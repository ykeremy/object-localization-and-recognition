# Preprocessing of images
import torch


class Padding(object):
    """
    Pad the images to match the desired output size.
    Args:
    output_size : Desired output size.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, list))
        self.output_size = output_size

    def __call__(self, image):  # image is tensor, before calling Pad, call ToTensor()
        padded_image = torch.zeros([3, *self.output_size])
        image_size = (
            image.size()
        )  # get the resized image size to place the original image in the padded image
        print(image_size)
        if image_size[1] > image_size[2]:  # width is 224 pixels, pad height
            starting_index = int((self.output_size[1] - image_size[2]) / 2)
            finishing_index = starting_index + image_size[2]
            padded_image[:, :, starting_index:finishing_index] = image
        else:  # height is 224 pixels, pad width
            starting_index = int((self.output_size[0] - image_size[1]) / 2)
            finishing_index = starting_index + image_size[1]
            padded_image[:, starting_index:finishing_index, :] = image

        return padded_image


class Resize(object):
    def __init__(self, desired_dimension):
        assert isinstance(desired_dimension, int)
        self.desired_dimension = desired_dimension

    def __call__(self, image):
        width, height = image.size
        aspect_ratio = width / height

        if width > height:
            new_width = self.desired_dimension
            new_height = int(self.desired_dimension / aspect_ratio)
        else:
            new_height = self.desired_dimension
            new_width = int(self.desired_dimension * aspect_ratio)

        resized_image = image.resize((new_width, new_height), resample=0)

        return resized_image


class Normalize(object):
    def __init__(self, mean, std):  # mean and std are arrays
        self.mean = mean
        self.std = std

    def __call__(self, image):
        normalized_image = torch.zeros(image.size())
        r = image[0, :, :]
        normalized_r = (r - self.mean[0]) / self.std[0]  # mean 0, std 1
        normalized_image[0, :, :] = normalized_r

        g = image[1, :, :]
        normalized_g = (g - self.mean[1]) / self.std[1]  # mean 0, std 1
        normalized_image[1, :, :] = normalized_g

        b = image[2, :, :]
        normalized_b = (b - self.mean[2]) / self.std[2]  # mean 0, std 1
        normalized_image[2, :, :] = normalized_b

        return normalized_image
