# coding utf-8


from GMM import EM_algorithm

# input : audio files directories
global audio_file_list_dir

audio_file_list_dir = {'Antonio': None, 'Helene': None,
                       'Leo': None, 'Mathilde': None, 'Nigil': None}

# define features extraction processing


def feat_extract(audio_file=None):
    # unfinished business
    return(audio_file)


# define class to store info
class personal_data:

    def __init__(self, name):
        self.name = name
        self.audio_files = None
        self.audio_file_list_dir = audio_file_list_dir[name]
        self.features_list = None

    def add_data(self):
        for audio_file in self.audio_file_list_dir:
            self.audio_files += [audio_file]

    def extract_features(self):
        for audio_file in self.audio_files:
            self.features_list += [feat_extract(audio_file)]

# create classes
antonio_data = personal_data('Antonio')
helene_data = personal_data('Helene')
leo_data = personal_data('Leo')
mathilde_data = personal_data('Mathilde')
nigil_data = personal_data('Nigil')

data_list = [antonio_data, helene_data, leo_data, mathilde_data, nigil_data]

# extract features and build samples list
S = []
for data_element in data_list:
    data_element.add_data()
    data_element.extract_features()
    S += [data_element.features_list]


# parameter estimation : EM algorithm
model = EM_algorithm(S=S, N=len(S), K=len(S[0]))

# print parameters and visualize gaussians
print(model.get_parameters)
# unfinished business
