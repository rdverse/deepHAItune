from tensorflow.keras import layers
import tensorflow as tf
import yaml

class parseConf:
    def __init__(self):
        configs = getConfig()
            
    def getConfig(self):
        yamlPath = 'config/model1.yaml'
        file = open(yamlPath)
        return(yaml(file))
    

class layerzoo:
    '''
    class :layerzoo
    -----  
    arguments: 1) layer : Name of the layer - get the layer 
    ---------- 2) config: Dictionary of all layer feature values
               3) prev:   Layer until previous point
               4) hp:     hyperparameter tuning argument
'''
    
    def __init__(self, layer,config, model,hp):

        buildLayer = getattr(self, layer, lambda : "Layer has to be defined")
        return(buildLayer(config))
    
    
    def Dense(self, config, prev = model, hp = hp):
        
        if config['tune']:
            layer = layers.Dense(min_value = config['units_min'],\
                                 max_value = config['units_max'],\
                                 step = config['step'],\
                                 activation = config['activation'])(prev)
        else:
            layer = layers.Dense(units = config['units'],\
                                 activation = config['activation'])(prev)
        return(layer)
    
    
    def Conv2D(self, prev, config,hp =None):
        
        if config['tune']:
            layer = layers.Conv2D(filters = config['filters'],\
                                  kernel_size = config['kernel_size'],\
                                  padding = config['padding'],\
                                  activation = config['activation'])
        else:
            layer = layers.Conv2D(filters = config['filters'],\
                                  kernel_size = config['kernel_size'],\
                                  padding = config['padding'],\
                                  activation = config['activation'])


        return(layer)
    


'''
Function : build_model
--------
args : #To be added , fetch the config file from main
----
Returns: Model 
------
'''  

def build_model(hp):
    configs = parseConf().configs
    
    input = layers.Input(shape = (3,150,1))
    model = input
    for layer, args in configs:
         model = layerzoo(layer,config,model,hp) 

    output = layers.Dense(1, activation='relu')(model)

    model = tf.keras.Model(inputs = input, outputs = output)
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(optimizer = optimizer,\
                                   loss='mean_absolute_error',\
                                   metrics=['mae','mape'])
    
     # optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),\
    return model
    




def build_and_tune_model():
             
