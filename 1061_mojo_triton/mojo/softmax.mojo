from sys import exit
from sys.info import has_accelerator

from python_interop import parse_softmax_args


def main():
    @parameter
    if not has_accelerator():
        print("No GPU detected.")
        exit(1)

    try:
        args = parse_softmax_args()
        for shape in args.shape:
            print("shape: (" + String(shape[0]) + ", " + String(shape[1]) + ")")
        print("runs: " + String(args.runs))
        print("save_tensors: " + String(args.save_tensors))
        print("verbose: " + String(args.verbose))
    except:
        exit(1)
