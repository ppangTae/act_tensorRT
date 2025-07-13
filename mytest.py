import onnx
onnx_model = onnx.load("policy.onnx")
onnx.checker.check_model(onnx_model)
