# ONNX model exporters

Scripts included:

- `export_vit_detector.py`: YOLOS, LW-DETR and RF-DETR from Hugging Face.
- `export_resnet.py`: torchvision ResNet/ResNeXt/Wide-ResNet models.
- `export_mobilenet.py`: torchvision MobileNet models.
- `export_yolo.py`: Ultralytics YOLO checkpoints.
- `export_vit.py`: torchvision ViT or Hugging Face ViT classifiers.
- `export_bert.py`: Hugging Face BERT base, masked-LM or sequence-classification models.

The default output directory is `../models`, resolved relative to these scripts. Every script accepts `--output-dir` to override it.

## Pipeline

Each script performs the following steps locally:

1. Download the original PyTorch weights.
2. Put the model in evaluation mode and, for Hugging Face transformer models, request eager attention when supported.
3. Export a standard ONNX graph with fixed dimensions unless dynamic axes are explicitly enabled.
4. Optionally convert the exported graph to FP16 or INT8.
5. Convert the default ONNX domain to the requested final opset with `onnx.version_converter.convert_version`. When `--opset` is omitted, the latest opset exposed by the installed `onnx` package is used.
6. Validate the final model with `onnx.checker`.

No ONNX Runtime graph optimization is applied by these scripts. In particular, they do not:

- create an `InferenceSession` to serialize an optimized graph;
- run ORT Basic, Extended or All graph optimizations;
- run the ONNX Runtime transformer optimizer;
- replace transformer subgraphs with fused ORT operators such as `Attention`, `Gelu` or `SkipLayerNormalization`.

The FP32 output is intentionally kept as a standard, provider-independent ONNX graph so that ORT optimization can be applied later, once the target execution provider and hardware are known. Transformer exports use eager attention and the trace-based PyTorch exporter to retain decomposed attention patterns that can be recognized by later ORT transformer fusion passes.

The intermediate export opset is capped at 18 for broad exporter compatibility. The final model is then converted to the requested opset.

## Precision variants

- `fp32`: unquantized standard ONNX graph. This is the preferred input for later ORT graph and transformer optimization.
- `fp16`: ONNX float16 conversion only; no ORT graph optimization.
- `int8`: ONNX Runtime's quantization API is used, but no ORT graph optimization is run. CNN and YOLO models use static QDQ quantization with calibration images; BERT and ViT-like models use dynamic weight quantization.

Quantization changes the graph structure and may prevent some transformer fusions that are available on the FP32 graph. When the workflow requires transformer fusion followed by quantization, export FP32 here, apply the desired ORT optimization later, and quantize in that later target-specific stage.

## Installation

```bash
python -m pip install -r requirements.txt
```

`onnxruntime` is only used by the optional INT8 quantization paths. FP32 and FP16 export do not instantiate an ORT inference session.

## Examples

```bash
python export_resnet.py --model resnet50 --dynamic-batch
python export_mobilenet.py --model mobilenet_v3_small --precision fp16
python export_vit.py --model vit_b_16
python export_vit.py --source huggingface --model google/vit-base-patch16-224
python export_vit_detector.py --family yolos --size tiny
python export_vit_detector.py --family lwdetr --size small --batch-size 1
python export_vit_detector.py --family rfdetr --size nano --model Roboflow/rf-detr-nano
python export_bert.py --model google-bert/bert-base-uncased --dynamic-batch --dynamic-sequence
python export_yolo.py --version yolo11 --size n --dynamic
```

Static INT8 export for CNNs and YOLO requires representative calibration images:

```bash
python export_resnet.py \
  --model resnet50 \
  --precision int8 \
  --calibration-dir ./calibration_images

python export_yolo.py \
  --version yolo11 \
  --size n \
  --precision int8 \
  --calibration-dir ./calibration_images
```
