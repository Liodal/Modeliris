#Libraries
import streamlit as st
import pandas as pd 
import seaborn as sns
from PIL import Image
import numpy as np
import plotly 
import plotly.express as px
import plotly.offline as pyo
from plotly.offline import init_notebook_mode,plot,iplot
import cufflinks as cf
import matplotlib.pyplot as plt


st.set_option('deprecation.showPyplotGlobalUse', False)


#Background color setup
st.markdown("""
<style>
body {
  background: #adadad; 
}
</style>
    """, unsafe_allow_html=True)


#Titre 
st.title("Projet IRIS Machine Learning")


#Image upload : image 1
image = Image.open('iris1.png')
st.image(image, caption='',use_column_width=True)


st.header("Buitl with Streamlit")


#Some information
st.header("Some information about iris flower")


st.subheader("""Iris, (genus Iris), genus of about 300 species of plants in the family Iridaceae, including some of the world’s most popular and varied garden flowers.The diversity of the genus is centred in the north temperate zone, though some of its most handsome species are native to the Mediterranean and central Asian areas. The iris is (arguably) the fleur-de-lis of the French royalist standard. It is a popular subject of Japanese flower arrangement, and it is also the source of orrisroot, from which “essence of violet” perfume is made.""")

st.header("Iris Identification by Flower Characteristics")
st.subheader("Most iris have similar looking flowers but there are a couple of unique characteristics that are used to identify different types of iris with rhizomes and these include the ‘beard’ and the ‘crest’.")

st.subheader("The aim of this model is to classify iris flowers among three species (setosa, versicolor, or virginica) from measurements of sepals and petals length and width.")


#Image upload : image 2
image2 = Image.open('iris2.png')
st.image(image2, caption='',use_column_width=True)


st.sidebar.title("Data Visualisation")


#File loading
st.sidebar.info("Only a iris.csv file can be used. \nDownload it on Kaggle web site https://www.kaggle.com/uciml/iris")
my_dataset = st.sidebar.file_uploader("Upload your iris file", type="csv")


#Load dataset
if my_dataset is not None :
    def explore_data(dataset):
        df = pd.read_csv(dataset)
        return df


    data = explore_data(my_dataset)


    if data.columns[5] == "Species":
        #Preview dataset
        if st.sidebar.checkbox("Preview Dataset"):
            if st.sidebar.button("Head"):
                st.header("Dataset Head Preview")
                st.write(data.head())

            elif st.sidebar.button("Tail"):
                st.header("Dataset Tail Preview")
                st.write(data.tail())

            else:
                st.write("Select head or Tail to preview dataset")


        #show all dataset
        if st.sidebar.checkbox("Show All Dataset"):
            st.header("All data")
            st.dataframe(data)


        #Show columns Name
        if st.sidebar.checkbox("Show Columns Names"):
            st.header("Columns Names")
            st.write(data.columns)


        #show Dimensions
        data_dim = st.sidebar.radio ("What dimensions do you want to see ?",("Rows","Columbs","All"))

        if st.sidebar.button("Show Dimensions"):
            if data_dim == 'Rows':
                st.write("Showing Rows : ",data.shape[0])

            elif data_dim == 'Columbs':
                st.write("Showing Columbs : ",data.shape[1])
            
            else:
                st.write("Showing Shape of Dataset : ",data.shape)
            

        #Show Summary
        if st.sidebar.checkbox("Show Summary of Dataset"):
            st.header("Dataset Information")
            st.write(data.describe())


        #Select A columns
        col_option = st.sidebar.selectbox("Select Column",data.columns)
        def affich (exemple):
            af = data[exemple]
            return af


        if st.sidebar.button("Visualize"): 
            st.header(col_option)  
            st.write(data[col_option])    
        

        #----------------------------------------Plot-------------------------------------


        #let's start data cleaning
        st.sidebar.title("Model Predict")


        #Let's rename our columns
        data.rename(columns={'SepalLengthCm':'SepalLength','SepalWidthCm':'SepalWidth','PetalLengthCm':'PetalLength','PetalWidthCm':'PetalWidth'},inplace=True)


        fig_option = st.sidebar.multiselect("Select Figure",("Plot1","Plot2","Plot3","Plot4","Plot5","Plot6","Plot7"))

        #Graphs plotting
        if st.sidebar.button("Show Plot"):

            for test in fig_option:
                #Species ploting with petal and sepal metrics
                if test == 'Plot1':
                    sns.set(style="whitegrid")
                    plt.figure(figsize=(12,10))
                    plt.subplot(2,2,1)
                    sns.violinplot(x="Species",y="SepalLength",data=data)
                    plt.subplot(2,2,2)
                    sns.violinplot(x="Species",y="SepalWidth",data=data)
                    plt.subplot(2,2,3)
                    sns.violinplot(x="Species",y="PetalLength",data=data)
                    plt.subplot(2,2,4)
                    sns.violinplot(x="Species",y="PetalWidth",data=data)
                    st.pyplot()   
                    st.header('Species ploting with petal and sepal metrics')

                #Petalwidth
                elif test == 'Plot2':
                    plotlyt_fig2 = px.scatter(data,x='Species',y='PetalWidth',size='PetalWidth',color='PetalWidth')
                    st.plotly_chart(plotlyt_fig2)
                    st.header('Species with petalwidth')

                #PetalLength
                elif test == 'Plot3':
                    sns.FacetGrid(data, hue="Species", height=5) \
                    .map(sns.distplot, "PetalLength") \
                    .add_legend();
                    st.pyplot()
                    st.header('Species with petalLength')

                #PetalWidth linear plot
                elif test == 'Plot4':
                    plotlyt_plot4 = px.line(data,x='Species',y='PetalWidth')
                    st.plotly_chart(plotlyt_plot4)
                    st.header('Species with petalwidth (linear plot)')

                #Display in matrix form according to Petals and Sepals
                elif test == 'Plot5':
                    plotlyt_plot5 = px.scatter_matrix(data,color='Species',title='Iris',dimensions=['SepalLength','SepalWidth','PetalWidth','PetalLength'])
                    st.plotly_chart(plotlyt_plot5)
                    st.header('Species, Petal and Sepal with matrix form')

                # SepalLength
                elif test == 'Plot6':
                        sns.FacetGrid(data, hue="Species", height=5) \
                        .map(sns.distplot, "SepalLength") \
                        .add_legend();
                        st.pyplot()   
                        st.header('Species with SepalLength')

                #SepalWidth
                elif test == 'Plot7':
                    plotlyt_fig2 = px.scatter(data,x='Species',y='SepalWidth',size='SepalWidth',color='SepalWidth')
                    st.plotly_chart(plotlyt_fig2)
                    st.header('Species with sepalwidth')
                else:
                    st.write("Select Column")


        st.sidebar.title("Decision Tree")
        #Let's initialize ours viriable to the model


        #Let's remove the species column
        x=data.drop(['Id','Species'],axis=1)
    

        #Let's initialize the second variable for training
        y=data['Species']

        var_view = st.sidebar.radio ("Let's view ours variables",("First variable : x","Second variable : y"))

        if st.sidebar.button("Show Variables"):    
            if var_view == 'First variable : x':
                st.header("Viriable x")
                st.write(x)

            elif var_view == 'Second variable : y':
                st.header("Viriable y")
                st.write(y)


        #Label encoding
        from sklearn.preprocessing import LabelEncoder


        LE = LabelEncoder()


        #Encode the String target features into integers
        y = LE.fit_transform(y)


        X=np.array(x)


        #training and testing
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=0)


        #Decesion tree
        from sklearn import tree
        from sklearn.metrics import accuracy_score

        DT=tree.DecisionTreeClassifier()

        DT.fit(X_train,y_train)

        prediction_DT=DT.predict(X_test)

        accuracy_DT=accuracy_score(y_test,prediction_DT)*100


        #Lets predict on custom input value
        Catagory=['Iris-Setosa','Iris-Versicolor','Iris-Virginica']

        X_DT=np.array([[5.1 ,3.5, 1.4, 0.2]])

        X_DT_prediction=DT.predict(X_DT)

        if st.sidebar.button("DT Testing") : 
            st.header("Decision Tree testing into dataset")
            st.write("Frist element : ",Catagory[int(X_DT_prediction[0])])

       
        if st.sidebar.button("Accuracy"):
            st.header("Decesion Tree")

            st.write("Accuracy : ",accuracy_DT)


        if st.sidebar.checkbox("Costum Prediction"):
            numb1 = st.number_input("SepalLength",min_value=0.0, max_value=8.0)
            numb2 = st.number_input("SepalWidth",min_value=0.0, max_value=8.0)
            numb3 = st.number_input("PetalLength",min_value=0.0, max_value=8.0)
            numb4 = st.number_input("PetalWidth",min_value=0.0, max_value=8.0)

            if st.button("Predict"):

                T_DT = np.array([[numb1 ,numb2, numb3, numb4]])

                T_DT_prediction=DT.predict(X_DT)

                st.write(Catagory[int(T_DT_prediction[0])])
    else:
        st.info("Please Choose a iris.csv file.")

#-----------------------------------------------------END-------------------------------------------------------------