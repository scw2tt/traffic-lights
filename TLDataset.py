from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
from data_helpers import get_label_and_path
import pyyaml as yaml

class TLDataset(VisionDataset):
    def __init__(self, root_dir_images, label_dir):

        # will be used by the get_item method
        self.root_dir_images = root_dir_images
        self.label_dir = label_dir
        self.list_of_labels = yaml.load(open(self.label_dir, 'rb').read())
        self.length = len(self.list_of_labels)

        # define the preprocessing for the images
        TLDataPreprocess = transforms.Compose(
            [
                transforms.ToTensor()
            ])
        self.transform = TLDataPreprocess
        return

    """
  needs to go through the yaml file and get the color.
  also, needs to go through and get the image of that id
  """

    def __getitem__(self, index):
        # function that gets the path,
        img_path, b_boxes, states = get_label_and_path(self.label_dir,
                                                       index,
                                                       self.list_of_labels)

        # add the root directory to the file path if it does not work correctly
        full_image_file_path = img_path.replace("/scratch/fs2/DTLD_final",
                                                self.root_dir_images)

        # grab the image from the path found in the converted yaml file
        img_pil = Image.open(full_image_file_path)
        img_pil = img_pil.resize((300, 300))
        # might need to remove the unsqueeze here
        image_data = self.transform(img_pil).unsqueeze(0)

        # encapsulate the b_boxes and states into a np array
        output_arr = [b_boxes, states]
        output_arr = np.array(output_arr)
        return image_data, output_arr

    def __len__(self):
        """The number of photos in the dataset."""
        return len(self.list_of_labels)
