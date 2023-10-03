
import pickle
import pandas as pd

#Load model
with open('knn_car.pkl', 'rb') as file:
    # Load the data from the file
                model, 
                buying_encoder, 
                maint_encoder, 
                doors_encoder, 
                persons_encoder, 
                lug_boot_encoder,
                safety_encoder,
                class_encode = pickle.load(file)

#Get New data
x_new = pd.DataFrame()
# Get user input for each variable
x_new['buying'] = [input('Enter buying [vhigh high med low ]: ')]
x_new['maint'] = [input('Enter maint [vhigh high med low ]: ')]
x_new['doors'] = [input('Enter doors [2 3 4 5more]: ')]
x_new['persons'] = [input('Enter persons [2 4 more]: ')]
x_new['lug_boot'] = [input('Enter lug_boot [small med big]: ')]
x_new['safety'] = [input('Enter safety [low med high]: ')]

#Encoding
x_new['buying'] = buying_encoder.transform(x_new['buying'])
x_new['maint'] = maint_encoder.transform(x_new['maint'])
x_new['doors'] = doors_encoder.transform(x_new['doors'])
x_new['persons'] = persons_encoder.transform(x_new['persons'])
x_new['lug_boot'] = lug_boot_encoder.transform(x_new['lug_boot'])
x_new['safety'] = safety_encoder.transform(x_new['safety'])

#Prediction
y_pred_new = model.predict(x_new)
result = class_encoder.inverse_transform(y_pred_new) 
print('Predicted class: ', result)
