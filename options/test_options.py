from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self._parser.add_argument('--input_path', type=str, help='path to image')
        self._parser.add_argument('--output_dir', type=str, default='./output', help='output path')
        self._parser.add_argument('--aus_csv_folder', type=str, default='./output', help='folder for reading aus from csv files')
        self._parser.add_argument('--au_index', type=int, default=0, help='which epoch to load? set to -1 to use latest cached model')
        self._parser.add_argument('--moods_pickle_file', type=str, default='/home/sevim/Downloads/master_thesis_study_documents/code-examples/affwild/annotations/105.pkl',
                    help='We are gonna select moods from this dataset of moods')
        self._parser.add_argument('--groundtruth_video', type=str, default='/home/sevim/Downloads/master_thesis_study_documents/code-examples/affwild/videos/105.avi', \
                    help='groundtruth video path')
        self._parser.add_argument('--comparison_model_name', type=str, default='affwild_per_frame_5frames', help = 'when two models are compared with same images')
        self._parser.add_argument('--comparison_load_epoch', type=int, default = -1, help = 'comparison models epoch number to load')
        self.is_train = False
