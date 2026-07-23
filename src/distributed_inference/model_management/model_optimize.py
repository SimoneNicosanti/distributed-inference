from __future__ import annotations

from pathlib import Path

import onnx
import onnxruntime as ort
import onnxruntime.transformers.optimizer as ort_transformers_opt
from onnxruntime.transformers.fusion_options import FusionOptions

from distributed_inference.domain.ModelGraphInfo import ModelInfo, ModelType

from enum import Enum


class OptimizationLevel(Enum):
    NONE = 0
    BASIC = 1
    EXTENDED = 2


def optimize_model(
    input_path: Path,
    output_path: Path,
    model_info: ModelInfo,
    opt_level: OptimizationLevel,
) -> None:

    optimize_with_ort(
        input_path,
        output_path,
        model_info,
        opt_level,
    )

    onnx.checker.check_model(output_path.as_posix())


def optimize_with_ort(
    input_path: Path,
    output_path: Path,
    model_info: ModelInfo,
    opt_level: OptimizationLevel,
) -> None:
    """
    Optimize the model using generic ORT optimizations or the
    Transformer Optimizer, depending on the requested level and model type.
    """

    match opt_level:
        case OptimizationLevel.BASIC | OptimizationLevel.NONE:
            _optimize_with_ort_standard(
                input_path=input_path,
                output_path=output_path,
                model_info=model_info,
                opt_level=opt_level,
            )

        case OptimizationLevel.EXTENDED:
            match model_info.type:
                case ModelType.CNN:
                    _optimize_with_ort_standard(
                        input_path=input_path,
                        output_path=output_path,
                        model_info=model_info,
                        opt_level=opt_level,
                    )

                case ModelType.VIT | ModelType.BERT:
                    _optimize_with_ort_transformer(
                        input_path=input_path,
                        output_path=output_path,
                        model_info=model_info,
                        opt_level=opt_level,
                    )

                    # _optimize_with_ort_standard(
                    #     input_path=input_path,
                    #     output_path=output_path,
                    #     model_info=model_info,
                    #     opt_level=opt_level,
                    # )

                case _:
                    raise ValueError(f"Unsupported model type: {model_info.type}")

        case _:
            raise ValueError(f"Unsupported optimization level: {opt_level}")

    if not output_path.is_file():
        raise RuntimeError("ONNX Runtime did not produce the optimized model")


def _optimize_with_ort_standard(
    input_path: Path,
    output_path: Path,
    model_info: ModelInfo,
    opt_level: OptimizationLevel,
) -> None:
    input_path = input_path.resolve(strict=True)
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    external_data_path = output_path.with_name(f"{output_path.name}.data")

    output_path.unlink(missing_ok=True)
    external_data_path.unlink(missing_ok=True)

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = _get_ort_opt_level(opt_level)
    session_options.optimized_model_filepath = str(output_path)

    match model_info.type:
        case ModelType.VIT | ModelType.BERT:
            ## We cannot do this optimization in the standard phase to avoid breaking the Attention pattern recognized by ort
            if opt_level == OptimizationLevel.BASIC:
                session_options.add_session_config_entry(
                    "optimization.disable_specified_optimizers",
                    "MatMulAddFusion",
                )
            else:
                ## We can do this optimization with the extended config but only after the ort transformer pass
                opt_level = OptimizationLevel.BASIC
                pass

        case _:
            pass

    # Il nome deve essere relativo alla directory del modello ONNX.
    session_options.add_session_config_entry(
        "session.optimized_model_external_initializers_file_name",
        external_data_path.name,
    )
    session_options.add_session_config_entry(
        "session.optimized_model_external_initializers_min_size_in_bytes",
        "1024",
    )

    ort.InferenceSession(
        str(input_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    if not output_path.is_file():
        raise RuntimeError(f"Optimized model not created: {output_path}")

    if not external_data_path.is_file():
        raise RuntimeError(f"External data not created: {external_data_path}")

    opt_onnx_model = onnx.load_model(
        str(output_path),
        load_external_data=True,
    )

    onnx.checker.check_model(opt_onnx_model)


def _optimize_with_ort_transformer(
    input_path: Path,
    output_path: Path,
    model_info: ModelInfo,
    opt_level: OptimizationLevel,
) -> None:
    """
    Apply ORT graph optimizations followed by Transformer-specific fusions.
    """
    model_type = _get_ort_transformer_model_type(model_info.type)

    options = FusionOptions(model_type=model_type)
    options = ort_transformers_opt.FusionOptions(
        model_type=model_type,
    )
    # options.enable_skip_layer_norm = False
    # options.enable_bias_skip_layer_norm = False

    optimized = ort_transformers_opt.optimize_model(
        input=input_path.as_posix(),
        model_type=model_type,
        num_heads=model_info.num_heads,
        hidden_size=model_info.hidden_size,
        optimization_options=options,
        opt_level=0,  ## Need to use this; otherwise it applies optimizations breaking the structure
        use_gpu=False,
        only_onnxruntime=False,
        verbose=True,
    )

    print(
        "Transformer fusion statistics:",
        optimized.get_fused_operator_statistics(),
    )

    # Keep all output data embedded. This avoids returning references to
    # files inside TemporaryDirectory.
    optimized.save_model_to_file(
        output_path.as_posix(),
    )


def _get_ort_transformer_model_type(
    model_type: ModelType,
) -> str:
    match model_type:
        case ModelType.BERT:
            return "bert"

        case ModelType.VIT:
            return "vit"

        case _:
            raise ValueError(f"Model type {model_type} is not a supported transformer")


def _get_ort_opt_level(
    opt_level: OptimizationLevel,
) -> ort.GraphOptimizationLevel:
    match opt_level:
        case OptimizationLevel.NONE:
            return ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        case OptimizationLevel.BASIC:
            return ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

        case OptimizationLevel.EXTENDED:
            return ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
