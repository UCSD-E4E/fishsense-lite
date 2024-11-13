import importlib
import random
import string

import numpy as np

from fishsense_lite.commands.label_studio_models.data import Data


# Adapted from https://github.com/HumanSignal/label-studio-converter/blob/master/label_studio_converter/brush.py
class SegmentationValue:
    def __init__(self, mask: np.ndarray):
        self.format = "rle"
        self.brushlabels = ["Fish"]
        self.rle = self.__mask2rle(self.__convert_mask(mask))

    def __convert_mask(self, mask: np.ndarray):
        unique_values = np.unique_values(mask[mask != 0])

        converted_mask = np.zeros_like(mask, dtype=np.uint8)
        for value in unique_values:
            converted_mask[mask == value] = 255

        return converted_mask

    def __mask2rle(self, mask: np.ndarray):
        """Convert mask to RLE

        :param mask: uint8 or int np.array mask with len(shape) == 2 like grayscale image
        :return: list of ints in RLE format
        """
        assert len(mask.shape) == 2, "mask must be 2D np.array"
        assert mask.dtype == np.uint8 or mask.dtype == int, "mask must be uint8 or int"
        array = mask.ravel()
        array = np.repeat(array, 4)  # must be 4 channels
        rle = self.__encode_rle(array)
        return rle

    def __encode_rle(self, arr: np.ndarray, wordsize=8, rle_sizes=[3, 4, 8, 16]):
        """Encode a 1d array to rle


        :param arr: flattened np.array from a 4d image (R, G, B, alpha)
        :type arr: np.array
        :param wordsize: wordsize bits for decoding, default is 8
        :type wordsize: int
        :param rle_sizes:  list of ints which state how long a series is of the same number
        :type rle_sizes: list
        :return rle: run length encoded array
        :type rle: list

        """
        # Set length of array in 32 bits
        num = len(arr)
        numbits = f"{num:032b}"

        # put in the wordsize in bits
        wordsizebits = f"{wordsize - 1:05b}"

        # put rle sizes in the bits
        rle_bits = "".join([f"{x - 1:04b}" for x in rle_sizes])

        # combine it into base string
        base_str = numbits + wordsizebits + rle_bits

        # start with creating the rle bite string
        out_str = ""
        for length_reeks, p, value in zip(*self.__base_rle_encode(arr)):
            # TODO: A nice to have but --> this can be optimized but works
            if length_reeks == 1:
                # we state with the first 0 that it has a length of 1
                out_str += "0"
                # We state now the index on the rle sizes
                out_str += "00"

                # the rle size value is 0 for an individual number
                out_str += "000"

                # put the value in a 8 bit string
                out_str += f"{value:08b}"
                state = "single_val"

            elif length_reeks > 1:
                state = "series"
                # rle size = 3
                if length_reeks <= 8:
                    # Starting with a 1 indicates that we have started a series
                    out_str += "1"

                    # index in rle size arr
                    out_str += "00"

                    # length of array to bits
                    out_str += f"{length_reeks - 1:03b}"

                    out_str += f"{value:08b}"

                # rle size = 4
                elif 8 < length_reeks <= 16:
                    # Starting with a 1 indicates that we have started a series
                    out_str += "1"
                    out_str += "01"

                    # length of array to bits
                    out_str += f"{length_reeks - 1:04b}"

                    out_str += f"{value:08b}"

                # rle size = 8
                elif 16 < length_reeks <= 256:
                    # Starting with a 1 indicates that we have started a series
                    out_str += "1"

                    out_str += "10"

                    # length of array to bits
                    out_str += f"{length_reeks - 1:08b}"

                    out_str += f"{value:08b}"

                # rle size = 16 or longer
                else:
                    length_temp = length_reeks
                    while length_temp > 2**16:
                        # Starting with a 1 indicates that we have started a series
                        out_str += "1"

                        out_str += "11"
                        out_str += f"{2 ** 16 - 1:016b}"

                        out_str += f"{value:08b}"
                        length_temp -= 2**16

                    # Starting with a 1 indicates that we have started a series
                    out_str += "1"

                    out_str += "11"
                    # length of array to bits
                    out_str += f"{length_temp - 1:016b}"

                    out_str += f"{value:08b}"

        # make sure that we have an 8 fold lenght otherwise add 0's at the end
        nzfill = 8 - len(base_str + out_str) % 8
        total_str = base_str + out_str
        total_str = total_str + nzfill * "0"

        rle = self.__bits2byte(total_str)

        return rle

    # Shamelessly plagiarized from https://stackoverflow.com/a/32681075/6051733
    def __base_rle_encode(self, inarray: np.ndarray):
        """run length encoding. Partial credit to R rle function.
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values)"""
        ia = np.asarray(inarray)  # force numpy
        n = len(ia)
        if n == 0:
            return None, None, None
        else:
            y = ia[1:] != ia[:-1]  # pairwise unequal (string safe)
            i = np.append(np.where(y), n - 1)  # must include last element posi
            z = np.diff(np.append(-1, i))  # run lengths
            p = np.cumsum(np.append(0, z))[:-1]  # positions
            return z, p, ia[i]

    def __bits2byte(self, arr_str, n=8):
        """Convert bits back to byte

        :param arr_str:  string with the bit array
        :type arr_str: str
        :param n: number of bits to separate the arr string into
        :type n: int
        :return rle:
        :type rle: list
        """
        rle = []
        numbers = [arr_str[i : i + n] for i in range(0, len(arr_str), n)]
        for i in numbers:
            rle.append(int(i, 2))
        return rle


class SegmentationResult:
    def __init__(self, mask: np.ndarray):
        height, width = mask.shape

        self.original_width = width
        self.original_height = height
        self.image_rotation = 0
        self.value = SegmentationValue(mask)

        letters_and_numbers = string.ascii_letters + string.digits

        self.id = "".join(random.choice(letters_and_numbers) for _ in range(10))
        self.from_name = "kp-1"
        self.to_name = "img-1"
        self.type = "keypointlabels"


class SegmentationPrediction:
    def __init__(self, mask: np.ndarray, model_name: str):
        self.model_version = (
            f"{model_name}.{importlib.metadata.version("pyfishsensedev")}"
        )
        self.result = [SegmentationResult(mask)]


class SegmentationLabelStudioJSON:
    def __init__(
        self,
        prefix: str,
        img: str,
        mask: np.ndarray,
        model_name: str,
    ):
        self.data = Data(prefix, img)
        self.predictions = (
            [SegmentationPrediction(mask, model_name)] if mask.sum() > 0 else []
        )
