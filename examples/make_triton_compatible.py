"""
This script makes the ONNX model compatible with Triton inference server.
"""

import sys
import numpy as np
import onnx
import onnxruntime as ort
import onnx_graphsurgeon as gs


def add_squeeze(graph, speed_input, speed_unsqueezed):
    """
    Add squeeze operation to the speed input to change shape from [batch_size, 1] to [batch_size]
    """
    # Create a squeeze node
    squeeze_node = gs.Node(
        op="Squeeze",
        name="speed_squeeze",
        inputs=[speed_unsqueezed],
        outputs=[gs.Variable(name="speed_squeezed", dtype=speed_unsqueezed.dtype)]
    )
    
    ## Find first node that has speed_unsqueezed as input
    insert_idx = 0
    for idx, node in enumerate(graph.nodes):
        for i, input_name in enumerate(node.inputs):
            if input_name.name == speed_unsqueezed.name:
                insert_idx = idx
                break
        if insert_idx != 0:
            break
    
    ## Add squeeze node to the graph
    insert_idx = min(0, insert_idx - 1)
    graph.nodes.insert(insert_idx, squeeze_node)
    
    # Update the speed input to point to the squeezed output
    for node in graph.nodes:
        for i, input_name in enumerate(node.inputs):
            if input_name.name == speed_input.name and not node.name == "speed_squeeze":
                node.inputs[i] = squeeze_node.outputs[0]
                
    return graph


def main():
    if len(sys.argv) != 2:
        print("Usage: python make_triton_compatible.py <onnx_model_path>")
        sys.exit(1)

    onnx_model_path = sys.argv[1]
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    print("Model is valid")
    
    graph = gs.import_onnx(onnx_model)

    ## get input_id for speed
    speed_idx, speed = None, None
    for idx, input_ in enumerate(graph.inputs):
        if input_.name=="speed":
            speed_idx = idx
            speed = input_

    # Update the speed input to have shape [batch_size, 1]
    speed_unsqueezed = gs.Variable(name="speed", dtype=speed.dtype, shape=[speed.shape[0], 1])
    graph.inputs[speed_idx] = speed_unsqueezed
        
    ## Add squeeze to change speed shape from [batch_size, 1] to [batch_size]
    if speed is not None:
        print(f"Found speed input: {speed.name}")
        print(f"Found speed input shape: {speed.shape}")
        print(f"Found speed input dtype: {speed.dtype}")
        print(f"Found speed input: {speed}")
        print(f"Found speed input: {type(speed)}")
        graph = add_squeeze(graph, speed, speed_unsqueezed)
        
        # Export the modified graph back to ONNX
        modified_model = gs.export_onnx(graph)
        onnx.checker.check_model(modified_model)
        
        # Save the modified model
        output_path = onnx_model_path.replace('.onnx', '_triton.onnx')
        onnx.save(modified_model, output_path)
        print(f"Modified model saved to: {output_path}")
    else:
        print("Speed input not found in the model")


if __name__ == "__main__":
    main()
