
import streamlit as st
import pickle
import pandas as pd

# Load the pre-trained model and encoders
def load_model():
    with open('knn_car.pkl', 'rb') as file:
        model, buying_encoder, maint_encoder, doors_encoder, persons_encoder, lug_boot_encoder, safety_encoder, class_encode = pickle.load(file)
    return model, buying_encoder, maint_encoder, doors_encoder, persons_encoder, lug_boot_encoder, safety_encoder, class_encode

model, buying_encoder, maint_encoder, doors_encoder, persons_encoder, lug_boot_encoder, safety_encoder, class_encode = load_model()

# Define a function to make predictions
def predict_class(buying, maint, doors, persons, lug_boot, safety):
    x_new = pd.DataFrame({
        'buying': [buying],
        'maint': [maint],
        'doors': [doors],
        'persons': [persons],
        'lug_boot': [lug_boot],
        'safety': [safety]
    })

    # Encoding
    x_new['buying'] = buying_encoder.transform(x_new['buying'])
    x_new['maint'] = maint_encoder.transform(x_new['maint'])
    x_new['doors'] = doors_encoder.transform(x_new['doors'])
    x_new['persons'] = persons_encoder.transform(x_new['persons'])
    x_new['lug_boot'] = lug_boot_encoder.transform(x_new['lug_boot'])
    x_new['safety'] = safety_encoder.transform(x_new['safety'])

    # Prediction
    y_pred_new = model.predict(x_new)
    result = class_encode.inverse_transform(y_pred_new)
    return result[0]

# Create a Streamlit app
st.title('Car Evaluation Prediction')

st.write("Enter the car attributes to predict its class.")

buying = st.selectbox('Buying Price:', ['vhigh', 'high', 'med', 'low'])
maint = st.selectbox('Maintenance Price:', ['vhigh', 'high', 'med', 'low'])
doors = st.selectbox('Number of Doors:', ['2', '3', '4', '5more'])
persons = st.selectbox('Number of Persons:', ['2', '4', 'more'])
lug_boot = st.selectbox('Luggage Boot Size:', ['small', 'med', 'big'])
safety = st.selectbox('Safety Level:', ['low', 'med', 'high'])

if st.button('Predict'):
    prediction = predict_class(buying, maint, doors, persons, lug_boot, safety)
    st.write(f'Predicted Class: {prediction}')
