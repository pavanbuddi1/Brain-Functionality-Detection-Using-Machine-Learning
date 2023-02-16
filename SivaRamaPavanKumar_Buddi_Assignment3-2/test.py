from classification import * 

folder = 0
imag = []
folder, imag = fun1(folder, imag)


for i in range(1,folder+1):
    tlf = pd.read_csv('./testPatient/Patient_{}_Labels.csv'.format(i))
    tlf.Label[tlf.Label==2] = 1
    tlf.Label[tlf.Label==3] = 1
    tlf.to_csv('./testPatient/newPatient_{}_Labels.csv'.format(i), index=False)


lab_indices_0 = []
lab_indices_1 = []
for i in range(1,folder+1):
    tlf = pd.read_csv('./testPatient/newPatient_{}_Labels.csv'.format(i))
    lab_count = imag[i-1]
    l_index_0 = []
    l_index_1 = []
    for j in range (0, lab_count):
        cValue = tlf.iloc[j]['Label']
        if(cValue==0):
            l_index_0.append(j+1)
        else:
            l_index_1.append(j+1)
    lab_indices_0.append(l_index_0)
    lab_indices_1.append(l_index_1)


trainDirectoryPath = os.path.join('./testPatient/','Train')
if(os.path.exists(trainDirectoryPath)):
    shutil.rmtree(trainDirectoryPath)
os.mkdir(trainDirectoryPath)


testDirectoryPath = os.path.join('./testPatient/','Test')
if(os.path.exists(testDirectoryPath)):
    shutil.rmtree(testDirectoryPath)
os.mkdir(testDirectoryPath)


validationDirectoryPath = os.path.join('./testPatient/','Validation')
if(os.path.exists(validationDirectoryPath)):
    shutil.rmtree(validationDirectoryPath)
os.mkdir(validationDirectoryPath)


train_value = folder - 2
test_value = train_value + 1
validationValue = test_value + 1


for i in range(1, train_value+1):
    trainPatientPath = os.path.join('./testPatient/Train','Patient_{}'.format(i))
    if(os.path.exists(trainPatientPath)):
        shutil.rmtree(trainPatientPath)
    os.mkdir(trainPatientPath)

    source = './testPatient/Patient_{}'.format(i)
    destination =  './testPatient/Train/Patient_{}'.format(i)
    for pngFile in glob.iglob(os.path.join(source, '*thresh.png')):
        shutil.copy(pngFile, destination)
    
    labelFilePath = './testPatient/newPatient_{}_Labels.csv'.format(i)
    shutil.copy(labelFilePath, destination)



testPatientPath = os.path.join('./testPatient/Test', 'Patient_{}'.format(test_value))
if(os.path.exists(testPatientPath)):
    shutil.rmtree(testPatientPath)
os.mkdir(testPatientPath)

source = './testPatient/Patient_{}'.format(test_value)
destination =  './testPatient/Test/Patient_{}'.format(test_value)
for pngFile in glob.iglob(os.path.join(source, '*thresh.png')):
    shutil.copy(pngFile, destination)

labelFilePath = './testPatient/newPatient_{}_Labels.csv'.format(test_value)
shutil.copy(labelFilePath, destination)


validationPatientPath = os.path.join('./testPatient/Validation','Patient_{}'.format(validationValue))
if(os.path.exists(validationPatientPath)):
    shutil.rmtree(validationPatientPath)
os.mkdir(validationPatientPath)

source = './testPatient/Patient_{}'.format(validationValue)
destination =  './testPatient/Validation/Patient_{}'.format(validationValue)
for pngFile in glob.iglob(os.path.join(source, '*thresh.png')):
    shutil.copy(pngFile, destination)

labelFilePath = './testPatient/newPatient_{}_Labels.csv'.format(validationValue)
shutil.copy(labelFilePath, destination)


trainPath = 'testPatient/Train'
testPath = 'testPatient/Test'
validationPath = 'testPatient/Validation'


x_train=[]
x_test=[]
x_validation=[]
y_train = []
y_test = []
y_validation = []
def createDataset():
    for i in range(1, train_value+1):
        imgFileCount = imag[i-1]
        for j in range(1, imgFileCount+1):
            imagePath = 'testPatient/Train/Patient_{}/IC_{}_thresh.png'.format(i, j)
            imgArray = cv.imread(imagePath)
            imgArray=cv.resize(imgArray,(224,224))
            x_train.append(imgArray)


    imgFileCount = imag[test_value-1]
    for j in range(1, imgFileCount+1):
        imagePath = 'testPatient/Test/Patient_{}/IC_{}_thresh.png'.format(test_value, j)
        imgArray = cv.imread(imagePath)
        imgArray=cv.resize(imgArray,(224,224))
        x_test.append(imgArray)


    imgFileCount = imag[validationValue-1]
    for j in range(1, imgFileCount+1):
        imagePath = 'testPatient/Validation/Patient_{}/IC_{}_thresh.png'.format(validationValue, j)
        imgArray = cv.imread(imagePath)
        imgArray=cv.resize(imgArray,(224,224))
        x_validation.append(imgArray)

    train_x = np.array(x_train)/255.0
    test_x = np.array(x_test)/255.0
    validation_x = np.array(x_validation)/255.0


    for i in range(1, train_value+1):
        yTrainTemp = pd.read_csv('./testPatient/Train/Patient_{}/newPatient_{}_Labels.csv'.format(i,i))
        t = yTrainTemp.Label.values
        for i in range(0, len(t)):
            y_train.append(t[i])


    yTestTemp = pd.read_csv('./testPatient/Test/Patient_{}/newPatient_{}_Labels.csv'.format(test_value,test_value))
    t = yTestTemp.Label.values
    for i in range(0, len(t)):
        y_test.append(t[i])


    yValidationTemp = pd.read_csv('./testPatient/Validation/Patient_{}/newPatient_{}_Labels.csv'.format(validationValue,validationValue))
    t = yValidationTemp.Label.values
    for i in range(0, len(t)):
        y_validation.append(t[i])
    
    return train_x, test_x, validation_x, y_train, y_test, y_validation

train_x, test_x, validation_x, y_train, y_test, y_validation = createDataset()
x_pred, y_pred, valid_pred = modelLoad(train_x, test_x, validation_x, y_train, y_test, y_validation)
create_results(folder, imag, x_pred, y_pred, valid_pred, test_value, validationValue)