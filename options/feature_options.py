from .base_options import BaseOptions


class FeatureOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        #self._parser.add_argument('--input_path', type=str, help='path to image')
        self._parser.add_argument('--output_dir', type=str, default='./output', help='output path')
	self._parser.add_argument('--images_folder', type=str, default='imgs', help='images folder')
        self._parser.add_argument('--output_file', type=str, default='./output', help='output file to keep the features')
        self._parser.add_argument('--input_file', type=str, default='./output', help='the path for the file which keeps the names of images')
        self._parser.add_argument('--input_folder', type=str, default='./output', help='folder where images are stored')
        self._parser.add_argument('--aus_file', type=str, default='./output', help='file where action unit values are stored')
        self.is_train = False
