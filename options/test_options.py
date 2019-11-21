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
        self.is_train = False


    def parse(self):
        if not self._initialized:
            self.initialize()
        self._opt = self._parser.parse_args()

        # set is train or set
        self._opt.is_train = self.is_train

        # set and check load_epoch
        self._opt.load_epoch = self._set_and_check_load_epoch(self._opt.name, self._opt.load_epoch)
        # get and set gpus
        self._get_set_gpus()

        args = vars(self._opt)

        # print in terminal args
        self._print(args)

        # save args to file
        self._save(args)

        return self._opt
