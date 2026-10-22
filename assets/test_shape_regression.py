import onnx_tool


def resnet18():
    onnxfile = "../smollm2_attention_dynamic_opt.onnx"
    model = onnx_tool.Model(onnxfile)
    graph = model.graph
    comp_graph = graph.get_compute_graph()

    comp_graph.save_model("resnet50_comp_graph.onnx")

    # # try update shape
    # print_tensor = "resnetv15_stage4__plus1"
    # tm = timer()
    # count = 0
    # for b in range(1, 5, 1):
    #     shapeengine.update_variable("batch", b)  # update the batch size
    #     for j in range(224, 299, 8):
    #         shapeengine.update_variable("h", j)  # update the height of input image
    #         for i in range(224, 257, 16):
    #             shapeengine.update_variable("w", i)  # update the width of input image
    #             shapeengine.update_variables()  # update all shape variables
    #             for tensor in graph.dynamics:
    #                 shape = shapeengine.get_tensorshape(tensor)  # query tensor shapes
    #                 if tensor == print_tensor:
    #                     print(b, j, i, shape)
    #             count += 1
    # t = tm.stop()
    # print(f"Total:{t} Time per reshape:{t / count}")  # less than 1us


if __name__ == "__main__":
    resnet18()
