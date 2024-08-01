import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model
model_path = 'D:\AI_Project\Student_Performance_Analysis\ML_MODEL\knn_model.pkl'
model = joblib.load(model_path)

# Extract feature names from the model if they are available
try:
    expected_columns = model.feature_names_in_
except AttributeError:
    st.error("The model does not contain feature names. Please ensure the model is trained with feature names.")

def main():
    # Set the title of the web app
    st.title('Student Pass/Fail Prediction')

    # Add a description
    st.write('Enter student information to predict pass or fail.')

    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader('Student Information')

        # Add input fields for features
        student_name = st.text_input('Student Name')
        sex = st.selectbox("Student's Sex", ['Female', 'Male'])
        age = st.slider("Student's Age", 15, 22, 17)
        address = st.selectbox("Home Address Type", ['Urban', 'Rural'])
        famsize = st.selectbox("Family Size", ['3 or Less', 'More than 3'])
        Pstatus = st.selectbox("Parents' Cohabitation Status", ['Living Together', 'Apart'])
        Medu = st.slider("Mother's Education Level", 0, 4, 2)
        Fedu = st.slider("Father's Education Level", 0, 4, 2)
        Mjob = st.selectbox("Mother's Job", ['Teacher', 'Healthcare', 'Services', 'At Home', 'Other'])
        Fjob = st.selectbox("Father's Job", ['Teacher', 'Healthcare', 'Services', 'At Home', 'Other'])
        reason = st.selectbox("Reason for Choosing This School", ['Close to Home', 'School Reputation', 'Course Preference', 'Other'])
        guardian = st.selectbox("Student's Guardian", ['Mother', 'Father', 'Other'])
        traveltime = st.slider("Travel Time to School (minutes)", 1, 4, 2)
        studytime = st.slider("Weekly Study Time (hours)", 1, 4, 2)
        failures = st.slider("Number of Past Class Failures", 0, 3, 0)
        schoolsup = st.selectbox("Extra Educational Support", ['No', 'Yes'])
        famsup = st.selectbox("Family Educational Support", ['No', 'Yes'])
        paid = st.selectbox("Extra Paid Classes", ['No', 'Yes'])
        activities = st.selectbox("Extra-curricular Activities", ['No', 'Yes'])
        nursery = st.selectbox("Attended Nursery School", ['No', 'Yes'])
        higher_edu = st.selectbox("Wants Higher Education", ['No', 'Yes'])
        internet = st.selectbox("Internet Access at Home", ['No', 'Yes'])
        romantic = st.selectbox("In a Romantic Relationship", ['No', 'Yes'])
        famrel = st.slider("Family Relationship Quality", 1, 5, 3)
        freetime = st.slider("Free Time After School", 1, 5, 3)
        goout = st.slider("Going Out with Friends", 1, 5, 3)
        Dalc = st.slider("Workday Alcohol Consumption", 1, 5, 3)
        Walc = st.slider("Weekend Alcohol Consumption", 1, 5, 3)
        health = st.slider("Current Health Status", 1, 5, 3)
        G3 = st.slider('Final Grade (1-20)', 1, 20, 10)
        GPA = st.slider('Grade Point Average (0-4)', 0.0, 4.0, 2.0)
        absences = st.slider('Number of School Absences', 0, 50, 10)
        untrained_column = st.text_input('Additional Information (not used in prediction)')

    # Convert categorical inputs to numerical
    sex = 1 if sex == 'Female' else 0
    address = 1 if address == 'Urban' else 0
    famsize = 1 if famsize == '3 or Less' else 0
    Pstatus = 1 if Pstatus == 'Living Together' else 0
    Mjob = {'At Home': 0, 'Healthcare': 1, 'Other': 2, 'Services': 3, 'Teacher': 4}.get(Mjob, 0)
    Fjob = {'At Home': 0, 'Healthcare': 1, 'Other': 2, 'Services': 3, 'Teacher': 4}.get(Fjob, 0)
    reason = {'Close to Home': 0, 'Course Preference': 1, 'Other': 2, 'School Reputation': 3}.get(reason, 0)
    guardian = {'Father': 0, 'Mother': 1, 'Other': 2}.get(guardian, 0)
    schoolsup = 1 if schoolsup == 'Yes' else 0
    famsup = 1 if famsup == 'Yes' else 0
    paid = 1 if paid == 'Yes' else 0
    activities = 1 if activities == 'Yes' else 0
    nursery = 1 if nursery == 'Yes' else 0
    higher = 1 if higher_edu == 'Yes' else 0
    internet = 1 if internet == 'Yes' else 0
    romantic = 1 if romantic == 'Yes' else 0

    # Prepare input data as a DataFrame
    input_data = pd.DataFrame({
        'sex_F': [sex],
        'sex_M': [1 - sex],
        'age': [age],
        'address_R': [1 - address],
        'address_U': [address],
        'famsize_GT3': [1 - famsize],
        'famsize_LE3': [famsize],
        'Pstatus_A': [1 - Pstatus],
        'Pstatus_T': [Pstatus],
        'Mjob_at_home': [1 if Mjob == 0 else 0],
        'Mjob_health': [1 if Mjob == 1 else 0],
        'Mjob_other': [1 if Mjob == 2 else 0],
        'Mjob_services': [1 if Mjob == 3 else 0],
        'Mjob_teacher': [1 if Mjob == 4 else 0],
        'Fjob_at_home': [1 if Fjob == 0 else 0],
        'Fjob_health': [1 if Fjob == 1 else 0],
        'Fjob_other': [1 if Fjob == 2 else 0],
        'Fjob_services': [1 if Fjob == 3 else 0],
        'Fjob_teacher': [1 if Fjob == 4 else 0],
        'reason_course': [1 if reason == 1 else 0],
        'reason_home': [1 if reason == 0 else 0],
        'reason_other': [1 if reason == 2 else 0],
        'reason_reputation': [1 if reason == 3 else 0],
        'guardian_father': [1 if guardian == 0 else 0],
        'guardian_mother': [1 if guardian == 1 else 0],
        'guardian_other': [1 if guardian == 2 else 0],
        'schoolsup_no': [1 - schoolsup],
        'schoolsup_yes': [schoolsup],
        'famsup_no': [1 - famsup],
        'famsup_yes': [famsup],
        'paid_no': [1 - paid],
        'paid_yes': [paid],
        'activities_no': [1 - activities],
        'activities_yes': [activities],
        'nursery_no': [1 - nursery],
        'nursery_yes': [nursery],
        'higher_no': [1 - higher],
        'higher_yes': [higher],
        'internet_no': [1 - internet],
        'internet_yes': [internet],
        'romantic_no': [1 - romantic],
        'romantic_yes': [romantic],
        'Medu': [Medu],
        'Fedu': [Fedu],
        'studytime': [studytime],
        'failures': [failures],
        'famrel': [famrel],
        'freetime': [freetime],
        'goout': [goout],
        'Dalc': [Dalc],
        'Walc': [Walc],
        'health': [health],
        'G3': [G3],
        'GPA': [GPA],
        'absences': [absences],
        'traveltime': [traveltime]
    })

    # Ensure columns are in the same order as during model training
    input_data = input_data[expected_columns]

    # Prediction and results section
    with col2:
        st.subheader('Prediction')
        if st.button('Predict'):
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)[0][1]
            
            st.write(f'Prediction for {student_name}: {"Pass" if prediction[0] == 1 else "Fail"}')
            st.write(f'Probability of Passing: {probability:.2f}')
            st.write(f'GPA: {GPA}')

            # Plotting
            fig, axes = plt.subplots(3, 1, figsize=(8, 16))

            # Plot Pass/Fail probability
            sns.barplot(x=['Fail', 'Pass'], y=[1 - probability, probability], ax=axes[0], palette=['red', 'green'])
            axes[0].set_title('Pass/Fail Probability')
            axes[0].set_ylabel('Probability')

            # Plot GPA distribution
            sns.histplot(input_data['GPA'], kde=True, ax=axes[1])
            axes[1].set_title('GPA Distribution')

            # Plot Pass/Fail pie chart
            axes[2].pie([1 - probability, probability], labels=['Fail', 'Pass'], autopct='%1.1f%%', colors=['red', 'green'])
            axes[2].set_title('Pass/Fail Pie Chart')

            # Display the plots
            st.pyplot(fig)

            # Provide recommendations
            if prediction[0] == 1:
                st.success(f"{student_name} is likely to pass. Keep up the good work!")
            else:
                st.error(f"{student_name} is likely to fail. Consider improving study habits and seeking additional help.")

if __name__ == '__main__':
    main()
