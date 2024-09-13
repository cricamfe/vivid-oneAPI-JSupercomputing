import cv2
import numpy as np
import sys

def write_raw_image(image, filename):
    try:
        with open(filename, 'wb') as file:
            rows, cols = image.shape
            depth = image.dtype
            channels = 1  # Grayscale image has only 1 channel

            file.write(np.int32(rows).tobytes())
            file.write(np.int32(cols).tobytes())
            file.write(np.int32(cv2.CV_32F).tobytes())  # Write the OpenCV depth constant
            file.write(np.int32(cv2.CV_32F).tobytes())  # Write the OpenCV type constant
            file.write(np.int32(channels).tobytes())

            size_in_bytes = image.nbytes
            file.write(np.int32(size_in_bytes).tobytes())
            file.write(image.tobytes())
    except Exception as e:
        print(f"Error while writing raw image: {e}")
        return False
    return True

def main(argv):
    if len(argv) < 3:
        print("Usage: python convert_image_to_bin.py <image_path> <output_bin_path>")
        return -1

    example_image_path = argv[1]

    example_image = cv2.imread(example_image_path, cv2.IMREAD_GRAYSCALE)

    if example_image is None:
        print("Could not open or find the image")
        return -1

    example_image = example_image.astype(np.float32)

    height, width = example_image.shape

    print(f"Width={width}; Height={height}")

    if write_raw_image(example_image, argv[2]):
        print("OK!")
        return 0
    else:
        print("Error!")
        return 1

if __name__ == "__main__":
    sys.exit(main(sys.argv))
