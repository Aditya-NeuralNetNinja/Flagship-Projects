# Exporting Stacking Regressor Model

    import pickle

    pickle.dump(df,open('df.pkl','wb'))
    pickle.dump(pipe,open('pipe.pkl','wb'))

    X_test.sample()

# Testing model on input example

[Screenshot (2276).jpg]

    input_example = [["Apple","Ultrabook",8,1.37,0,1,227,"Intel Core i5",1.8,0,128,"Intel","Mac"]]

    y_pred = pipe.predict(input_example)

    # predicted price = e^(predicted price *10)
    # taking inverse transform of (log and division by 10)

    predicted_price = np.exp(y_pred*10)
    print("The predicted price for your laptop is: Rs.",int(predicted_price))
