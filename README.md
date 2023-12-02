# auto_deploy
automated map the dl model on the fpga backend
environment:

    tvm, numpy=1.20, modelsim, torch=1.12 for onnx export


> Op Name为testbench名称，存在testbench_前缀；Relay Op为tvm注册的加速器算子名，存在relay.accel.vit.前缀

