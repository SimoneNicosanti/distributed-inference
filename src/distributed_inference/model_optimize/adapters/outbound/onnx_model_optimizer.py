from pathlib import Path

import onnx
import onnx.external_data_helper
import onnxruntime as ort
import onnxruntime.transformers.optimizer as ort_transformers_opt
from onnxruntime.transformers.fusion_options import FusionOptions
from typing_extensions import override

from distributed_inference.domain.model_graph_info import (
    ModelInfo,
    ModelType,
)
from distributed_inference.model_artifact.domain.artifact_bundle import (
    ArtifactConcretePaths,
)
from distributed_inference.model_optimize.application.ports.outbound.model_optimizer import (
    ModelOptimizer,
)
from distributed_inference.model_optimize.domain.optimization_level import (
    OptimizationLevel,
)


class OnnxModelOptimizer(ModelOptimizer):
    @override
    def optimize_model(
        self,
        input_paths: ArtifactConcretePaths,
        output_paths: ArtifactConcretePaths,
        model_info: ModelInfo,
        opt_level: OptimizationLevel,
    ) -> None:

        if input_paths.entrypoint_path is None:
            raise ValueError("Entrypoint path must be set when optimizing model")
        input_entrypoint_path = input_paths.entrypoint_path.resolve(strict=True)

        output_paths.root_path.mkdir(parents=True, exist_ok=True)
        output_entrypoint_path = output_paths.root_path.resolve(strict=True).joinpath(
            f"opt_level_{opt_level}.onnx"
        )

        match opt_level:
            case OptimizationLevel.BASIC | OptimizationLevel.NONE:
                self._optimize_with_ort_standard(
                    input_path=input_entrypoint_path,
                    output_path=output_entrypoint_path,
                    model_info=model_info,
                    opt_level=opt_level,
                )

            case OptimizationLevel.EXTENDED:
                match model_info.type:
                    case ModelType.CNN:
                        self._optimize_with_ort_standard(
                            input_path=input_entrypoint_path,
                            output_path=output_entrypoint_path,
                            model_info=model_info,
                            opt_level=opt_level,
                        )

                    case ModelType.VIT | ModelType.BERT:
                        self._optimize_with_ort_transformer(
                            input_path=input_entrypoint_path,
                            output_path=output_entrypoint_path,
                            model_info=model_info,
                            opt_level=opt_level,
                        )

                    case _:
                        raise ValueError(f"Unsupported model type: {model_info.type}")

            case _:
                raise ValueError(f"Unsupported optimization level: {opt_level}")

        if not output_entrypoint_path.is_file():
            raise RuntimeError("ONNX Runtime did not produce the optimized model")

        output_paths.entrypoint_path = output_entrypoint_path

    def _optimize_with_ort_standard(
        self,
        input_path: Path,
        output_path: Path,
        model_info: ModelInfo,
        opt_level: OptimizationLevel,
    ) -> None:

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = self._get_ort_opt_level(opt_level)
        session_options.optimized_model_filepath = str(output_path)

        ## Checking if the model uses external data
        uses_external_data = self._check_model_uses_external_data(input_path)

        external_data_path = output_path.with_name(f"{output_path.name}.data")
        external_data_path.unlink(missing_ok=True)
        if uses_external_data:
            ## If the original model uses external data we follow the same pattern

            # External data path must be with respect to the model path
            # We remove it if it already exists
            session_options.add_session_config_entry(
                "session.optimized_model_external_initializers_file_name",
                external_data_path.name,
            )

            ## We use this to externalize the initializers whatever the weights size is
            session_options.add_session_config_entry(
                "session.optimized_model_external_initializers_min_size_in_bytes",
                "0",
            )

        ort.InferenceSession(
            str(input_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

        if not output_path.is_file():
            raise RuntimeError(f"Optimized model not created: {output_path}")

        if uses_external_data:
            if not external_data_path.is_file():
                raise RuntimeError(
                    f"Optimized model external data not created: {external_data_path}"
                )

        onnx.checker.check_model(str(output_path))

    def _check_model_uses_external_data(self, input_path: Path) -> bool:
        uses_external_data = False
        for initializer_proto in onnx.load_model(input_path).graph.initializer:
            if onnx.external_data_helper.uses_external_data(initializer_proto):
                uses_external_data = True
                break
        return uses_external_data

    def _optimize_with_ort_transformer(
        self,
        input_path: Path,
        output_path: Path,
        model_info: ModelInfo,
        opt_level: OptimizationLevel,
    ) -> None:
        model_type = self._get_ort_transformer_model_type(model_info.type)

        options = FusionOptions(model_type=model_type)
        options = ort_transformers_opt.FusionOptions(
            model_type=model_type,
        )

        optimized = ort_transformers_opt.optimize_model(
            input=str(input_path),
            model_type=model_type,
            num_heads=model_info.num_heads,
            hidden_size=model_info.hidden_size,
            optimization_options=options,
            opt_level=0,  ## Need to use this; otherwise it applies optimizations breaking the structure
            use_gpu=False,
            only_onnxruntime=False,
            verbose=True,
        )

        use_external_data = self._check_model_uses_external_data(input_path)
        optimized.save_model_to_file(
            str(output_path),
            use_external_data_format=use_external_data,
        )

    def _get_ort_transformer_model_type(
        self,
        model_type: ModelType,
    ) -> str:
        match model_type:
            case ModelType.BERT:
                return "bert"

            case ModelType.VIT:
                return "vit"

            case _:
                raise ValueError(
                    f"Model type {model_type} is not a supported transformer"
                )

    def _get_ort_opt_level(
        self,
        opt_level: OptimizationLevel,
    ) -> ort.GraphOptimizationLevel:
        match opt_level:
            case OptimizationLevel.NONE:
                return ort.GraphOptimizationLevel.ORT_DISABLE_ALL

            case OptimizationLevel.BASIC:
                return ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

            case OptimizationLevel.EXTENDED:
                return ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
