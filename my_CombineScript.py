#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#
#                                   ES335- Machine Learning- Assignment 1
#
# This script combines the data from the UCI HAR Dataset into a more usable format.
# The data is combined into a single csv file for each subject and activity. 
# The data is then stored in the Combined folder.
#
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# Library imports
import pandas as pd
import numpy as np
import os

# Give the path of the test and train folder of UCI HAR Dataset
train_path = "./UCI HAR Dataset/train"
test_path = "./UCI HAR Dataset/test"

# Dictionary of activities. Provided by the dataset.
ACTIVITIES = {
    1: 'WALKING'            ,
    2: 'WALKING_UPSTAIRS'   ,
    3: 'WALKING_DOWNSTAIRS' ,
    4: 'SITTING'            ,
    5: 'STANDING'           ,
    6: 'LAYING'             ,
}

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                                        # Combining Traing Data
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# Loading the body_acc train data
body_acc_x = pd.read_csv(os.path.join(train_path,"Inertial Signals","body_acc_x_train.txt"),delim_whitespace=True,header=None)
body_acc_y = pd.read_csv(os.path.join(train_path,"Inertial Signals","body_acc_y_train.txt"),delim_whitespace=True,header=None)
body_acc_z = pd.read_csv(os.path.join(train_path,"Inertial Signals","body_acc_z_train.txt"),delim_whitespace=True,header=None)




# Read the subject IDs
subject_train = pd.read_csv(os.path.join(train_path,"subject_train.txt"),delim_whitespace=True,header=None)

# Read the labels
y = pd.read_csv(os.path.join(train_path,"y_train.txt"),delim_whitespace=True,header=None)


# Toggle through all the subjects.
for subject in np.unique(subject_train.values):

    sub_idxs = np.where( subject_train.iloc[:,0] == subject )[0]
    labels = y.loc[sub_idxs]

    # Toggle through all the labels.
    for label in np.unique(labels.values):

        # make the folder directory if it does not exist
        if not os.path.exists(os.path.join("Combined","Body_Train",ACTIVITIES[label])):
            os.makedirs(os.path.join("Combined","Body_Train",ACTIVITIES[label]))

        label_idxs = labels[labels.iloc[:,0] == label].index

        bodyx = []
        bodyy = []
        bodyz = []

        for idx in label_idxs:
            if bodyx is not None:
                bodyx = np.hstack((bodyx,body_acc_x.loc[idx][64:]))
                bodyy = np.hstack((bodyy,body_acc_y.loc[idx][64:]))
                bodyz = np.hstack((bodyz,body_acc_z.loc[idx][64:]))

            else:
                bodyx = body_acc_x.loc[idx]
                bodyy = body_acc_y.loc[idx]
                bodyz = body_acc_z.loc[idx]

        # saving the data into csv file
        data = pd.DataFrame({'bodyx':bodyx,'bodyy':bodyy,'bodyz':bodyz})
        save_path = os.path.join("Combined","Body_Train",ACTIVITIES[label],f"Subject_{subject}.csv")
        data.to_csv(save_path,index=False)

print("Done Combining the body acc training data")

# Loading the body_gyro train data
body_gyro_x = pd.read_csv(os.path.join(train_path,"Inertial Signals","body_gyro_x_train.txt"),delim_whitespace=True,header=None)
body_gyro_y = pd.read_csv(os.path.join(train_path,"Inertial Signals","body_gyro_y_train.txt"),delim_whitespace=True,header=None)
body_gyro_z = pd.read_csv(os.path.join(train_path,"Inertial Signals","body_gyro_z_train.txt"),delim_whitespace=True,header=None)

# Toggle through all the subjects.
for subject in np.unique(subject_train.values):

    sub_idxs = np.where( subject_train.iloc[:,0] == subject )[0]
    labels = y.loc[sub_idxs]

    # Toggle through all the labels.
    for label in np.unique(labels.values):

        # make the folder directory if it does not exist
        if not os.path.exists(os.path.join("Combined","Gyro_Train",ACTIVITIES[label])):
            os.makedirs(os.path.join("Combined","Gyro_Train",ACTIVITIES[label]))

        label_idxs = labels[labels.iloc[:,0] == label].index

        bodyx = []
        bodyy = []
        bodyz = []

        for idx in label_idxs:
            if bodyx is not None:
                bodyx = np.hstack((bodyx,body_gyro_x.loc[idx][64:]))
                bodyy = np.hstack((bodyy,body_gyro_y.loc[idx][64:]))
                bodyz = np.hstack((bodyz,body_gyro_z.loc[idx][64:]))

            else:
                bodyx = body_acc_x.loc[idx]
                bodyy = body_acc_y.loc[idx]
                bodyz = body_acc_z.loc[idx]

        # saving the data into csv file
        data = pd.DataFrame({'bodyx':bodyx,'bodyy':bodyy,'bodyz':bodyz})
        save_path = os.path.join("Combined","Gyro_Train",ACTIVITIES[label],f"Subject_{subject}.csv")
        data.to_csv(save_path,index=False)

print("Done Combining the gyro acc training data")

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                                        # Combining Test Data               
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# Loading the body_acc test data
body_acc_x = pd.read_csv(os.path.join(test_path,"Inertial Signals","body_acc_x_test.txt"),delim_whitespace=True,header=None)
body_acc_y = pd.read_csv(os.path.join(test_path,"Inertial Signals","body_acc_y_test.txt"),delim_whitespace=True,header=None)
body_acc_z = pd.read_csv(os.path.join(test_path,"Inertial Signals","body_acc_z_test.txt"),delim_whitespace=True,header=None)

# Read the subject IDs
subject_test = pd.read_csv(os.path.join(test_path,"subject_test.txt"),delim_whitespace=True,header=None)

# Read the labels
y = pd.read_csv(os.path.join(test_path,"y_test.txt"),delim_whitespace=True,header=None)

# Toggle through all the subjects.
for subject in np.unique(subject_test.values):
    
        sub_idxs = np.where( subject_test.iloc[:,0] == subject )[0]
        labels = y.loc[sub_idxs]

        # Toggle through all the labels.
        for label in np.unique(labels.values):
    
            if not os.path.exists(os.path.join("Combined","Body_Test",ACTIVITIES[label])):
                os.makedirs(os.path.join("Combined","Body_Test",ACTIVITIES[label]))
    
            label_idxs = labels[labels.iloc[:,0] == label].index
    
            bodyx = []
        bodyy = []
        bodyz = []

        for idx in label_idxs:
            if bodyx is not None:
                bodyx = np.hstack((bodyx,body_acc_x.loc[idx][64:]))
                bodyy = np.hstack((bodyy,body_acc_y.loc[idx][64:]))
                bodyz = np.hstack((bodyz,body_acc_z.loc[idx][64:]))

            else:
                bodyx = body_acc_x.loc[idx]
                bodyy = body_acc_y.loc[idx]
                bodyz = body_acc_z.loc[idx]
    
            # saving the data into csv file
            data = pd.DataFrame({'bodyx':bodyx,'bodyy':bodyy,'bodyz':bodyz})
        save_path = os.path.join("Combined","Body_Test",ACTIVITIES[label],f"Subject_{subject}.csv")
        data.to_csv(save_path,index=False)
        

print("Done Combining the body testing data")

# Loading the body_gyro test data
body_gyro_x = pd.read_csv(os.path.join(test_path,"Inertial Signals","body_gyro_x_test.txt"),delim_whitespace=True,header=None)
body_gyro_y = pd.read_csv(os.path.join(test_path,"Inertial Signals","body_gyro_y_test.txt"),delim_whitespace=True,header=None)
body_gyro_z = pd.read_csv(os.path.join(test_path,"Inertial Signals","body_gyro_z_test.txt"),delim_whitespace=True,header=None)

# Toggle through all the subjects.
for subject in np.unique(subject_test.values):
    
        sub_idxs = np.where( subject_test.iloc[:,0] == subject )[0]
        labels = y.loc[sub_idxs]

        # Toggle through all the labels.
        for label in np.unique(labels.values):
    
            if not os.path.exists(os.path.join("Combined","Gyro_Test",ACTIVITIES[label])):
                os.makedirs(os.path.join("Combined","Gyro_Test",ACTIVITIES[label]))
    
            label_idxs = labels[labels.iloc[:,0] == label].index
    
            bodyx = []
        bodyy = []
        bodyz = []

        for idx in label_idxs:
            if bodyx is not None:
                bodyx = np.hstack((bodyx,body_gyro_x.loc[idx][64:]))
                bodyy = np.hstack((bodyy,body_gyro_y.loc[idx][64:]))
                bodyz = np.hstack((bodyz,body_gyro_z.loc[idx][64:]))

            else:
                bodyx = body_gyro_x.loc[idx]
                bodyy = body_gyro_y.loc[idx]
                bodyz = body_gyro_z.loc[idx]
    
            # saving the data into csv file
            data = pd.DataFrame({'bodyx':bodyx,'bodyy':bodyy,'bodyz':bodyz})
        save_path = os.path.join("Combined","Gyro_Test",ACTIVITIES[label],f"Subject_{subject}.csv")
        data.to_csv(save_path,index=False)

print("Done Combining the gyro testing data")
print("done")
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=