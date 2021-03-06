config = {'train_data_path': ['/opt/LUNA16/subset0/',
                              '/opt/LUNA16/subset1/',
                              '/opt/LUNA16/subset2/',
                              '/opt/LUNA16/subset3/',
                              '/opt/LUNA16/subset4/',
                              '/opt/LUNA16/subset5/',
                              '/opt/LUNA16/subset6/',
                              '/opt/LUNA16/subset7/',
                              '/opt/LUNA16/subset8/'],
          'val_data_path': ['/opt/LUNA16/subset9/'],
          'test_data_path': ['/opt/LUNA16/subset9/'],

          'train_preprocess_result_path': '/opt/LUNAPreprocess/',
          # contains numpy for the data and label, which is generated by prepare.py
          'val_preprocess_result_path': '/opt/LUNAPreprocess/',
          # make sure copy all the numpy into one folder after prepare.py
          'test_preprocess_result_path': '/opt/LUNAPreprocess/',

          'train_annos_path': '/opt/LUNA16/CSVFILES/annotations.csv',
          'val_annos_path': '/opt/LUNA16/CSVFILES/annotations.csv',
          'test_annos_path': '/opt/LUNA16/CSVFILES/annotations.csv',

          'black_list': [],

          'preprocessing_backend': 'python',

          'luna_segment': '/opt/LUNA16/seg-lungs-LUNA16/',  # download from https://luna16.grand-challenge.org/data/
          'preprocess_result_path': '/opt/LUNAPreprocess/',
          'luna_data': '/opt/LUNA16/',
          'luna_label': '/opt/LUNA16/CSVFILES/annotations.csv'
          }
