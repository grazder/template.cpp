import sys

sys.path.append("../")

try:
    from build import bindings
except ImportError:
    raise ImportError("Please build package before running tests...")


def main():
    # Load model
    success, model = bindings.load_model("../ggml-model.bin")
    
    if success:
        print("Model loaded successfully")
    else:
        print("Failed to load model")
        return
    
    # Create example tensor
    input_data = [10, 10, 10, 10, 10]
    
    # Compute
    result = bindings.compute(model, input_data)
    
    # Print result
    print("Result:", result)

if __name__ == "__main__":
    main()
