from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self._parser.add_argument('--input_path', type=str, help='path to image')
        self._parser.add_argument('--output_dir', type=str, default='./output', help='output path')
        self._parser.add_argument('--aus_csv_folder', type=str, default='./output', help='folder for reading aus from csv files')
        self._parser.add_argument('--au_index', type=int, default=0, help='which epoch to load? set to -1 to use latest cached model')
        self.is_train = False
