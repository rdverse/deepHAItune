import yaml


file =  open('config/model1.yaml', 'r') 
    
yamldict = yaml.load(file)


print(type(yamldict))
print(yamldict)
