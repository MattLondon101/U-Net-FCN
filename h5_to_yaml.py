
from keras.models import model_from_yaml
dependencies = {
    'jaccard_index': jaccard_index
}

class hyam():
    def __init__(self,h5path,yamlpath):

        self.model=load_model(h5path,custom_objects=dependencies)

        m_yaml=self.model.to_yaml()
        with open(yamlpath,'w') as yf:
            yf.write(m_yaml)


# User input commands
h5path=str(input("Enter path to hdf5/h5 file: "))
yamlpath=str(input("Enter path to save yaml file: "))

# Initiate class
hyam(h5path,yamlpath)

# exec(open('h5_to_yaml.py').read())

