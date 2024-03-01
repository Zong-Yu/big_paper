from PIL import Image
#from streamlit_shap import st_shap
import streamlit as st
import numpy as np 
import pandas as pd 
import time
import plotly.express as px 
import seaborn as sns
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error,confusion_matrix,accuracy_score,recall_score,precision_score,classification_report,roc_auc_score
import shap
import lime
from lime import lime_tabular
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import pickle

plt.style.use('default')

st.set_page_config(
    page_title = 'ICU缺血性脑卒中院内死亡预测',
    page_icon = '🕵️‍♀️',
    layout = 'wide'
)

data = pd.read_csv("train_data.csv",index_col=False,header=0)
data=data.drop(data.columns[0], axis=1)
X = data.drop("outcome", axis=1)

# dashboard title
#st.title("Real-Time Fraud Detection Dashboard")
st.markdown("<h1 style='text-align: center; color: black;'>ICU缺血性脑卒中院内死亡预测</h1>", unsafe_allow_html=True)

# side-bar 

def user_input_features():
    st.sidebar.header('请在下方输入相关指标⬇️')
    a1 = st.sidebar.slider('年龄(岁)',min_value=18.0, max_value=100.0, value=60.0,step=0.1)
    a2 = st.sidebar.slider('红细胞压积(%)',min_value=10.0, max_value=70.0, value=30.0,step=0.1)
    a3 = st.sidebar.slider('收缩压(mmHg)', min_value=70.0, max_value=190.0, value=100.0,step=0.1)
    a4 = st.sidebar.selectbox("他汀类药物", ('是', '否'))
    a5 = st.sidebar.slider('尿素氮(mg/dL)', min_value=5.0, max_value=70.0, value=30.0,step=0.1)
    a6 = st.sidebar.slider('白细胞计数(10^9/L)', min_value=0.0, max_value=25.0, value=10.0,step=0.1)
    a7 = st.sidebar.selectbox("华法林", ('是', '否'))
    a8 = st.sidebar.selectbox("机械通气", ('是', '否'))
    a9 = st.sidebar.slider('碳酸氢盐(mEq/L)', min_value=10.0, max_value=40.0, value=23.0,step=0.1)
    if a4 == '是':
            a4 = 1
    else:
        a4 = 0
    if a7 == '是':
            a7 = 1
    else:
        a7 = 0
    if a8 == '是':
            a8 = 1
    else:
        a8 = 0

    output = [a1,a2,a3,a4,a5,a6,a7,a8,a9]
    return output


outputdf = user_input_features()

st.subheader('关于模型')
st.write('该模型的内部验证结果显示,其ROC曲线下面积(AUC)为 0.908(95% CI:0.882-0.933)，表明该模型具有很强的预测性能，校准曲线和决策曲线分析表明该模型具有良好的校准和临床收益。虽然该模型具有良好的预测性能，但必须注意的是，其使用应仅限于研究目的。这意味着该模型可用于在研究环境中获得洞察力、探索关系和提出假设。不过，在实际应用该模型之前，还需要进行更多的研究验证和严格评估。')

st.subheader('网页计算器指南')
st.write('计算器由三个主要部分组成:第一部分的左侧边栏允许用户输入相关参数并选择模型变量。第二部分显示院内死亡率的预测概率。第三部分提供了详细的模型信息,包括使用SHAP和LIME进行的全局和局部解释,为预测结果提供解释。希望本指南能帮助您有效利用我们的预测计算器。')

image4 = Image.open('shap.png')
shapdatadf =pd.read_excel(r'shapdatadf.xlsx')
shapvaluedf =pd.read_excel(r'shapvaluedf.xlsx')
# 这里是查看SHAP值

st.subheader('实时预测')
outputdf = pd.DataFrame([outputdf], columns= shapdatadf.columns)

#st.write('User input parameters below ⬇️')
#st.write(outputdf)
with open("RF1.pickle", "rb") as file:
    RF = pickle.load(file)

p1 = RF.predict(outputdf)[0]
p2 = RF.predict_proba(outputdf)
p2 = round(p2[0][1], 4)

#st.write('User input parameters below ⬇️')
#st.write(outputdf)
st.write(f'ICU缺血性脑卒中患者院内死亡概率: {p2}')
st.write(' ')

st.subheader("SHAP全局解释")
placeholder5 = st.empty()
with placeholder5.container():
    f1,f2 = st.columns(2)

    with f1:
        st.write('SHAP蜂群图')
        st.write(' ')
        st.image(image4)
        st.write('SHAP蜂群图旨在显示前几个重要变量对模型输出影响的摘要。给定解释中的每个样本都由每个变量行上的单个点表示，并且点会“堆积”在每个变量行上以显示密度。颜色用于显示变量的原始值（红色代表高值，蓝色代表低值）。')     
    with f2:
        st.write('SHAP依赖图')
        cf = st.selectbox("选择变量", (shapdatadf.columns))
        fig = px.scatter(x = shapdatadf[cf], 
                         y = shapvaluedf[cf], 
                         color=shapdatadf[cf],
                         color_continuous_scale= ['blue','red'],
                         labels={'x':'原始值', 'y':'SHAP值'})
        st.write(fig)  


# 这里是查看SHAP和LIME图像的
        
st.subheader("SHAP和LIME的局部解释")
placeholder6 = st.empty()
with placeholder6.container():
    f1,f2 = st.columns(2)
    with f1:
         explainer   = shap.TreeExplainer(RF)
         shap_values = explainer.shap_values(outputdf)
         plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
         plt.rcParams['axes.unicode_minus'] = False
         st.write('SHAP力图')
         st.set_option('deprecation.showPyplotGlobalUse', False)
         shap.force_plot(explainer.expected_value[1], shap_values[1][0,:],outputdf.iloc[0,:],link='logit',show=False, matplotlib=True)
         st.pyplot(bbox_inches='tight')
         st.write('上图显示了SHAP 力图可用于将每个变量的SHAP值可视化为一个力，它可以增加（正值）或减少（负值）相对于其基线的预测，用于对单个患者结局预测的解释。')
    with f2:
         explainer = lime.lime_tabular.LimeTabularExplainer(X.values, feature_names=X.columns.values.tolist(),
                                                  class_names=['生存',"死亡"], verbose=True, mode='classification',feature_selection='none')
         exp = explainer.explain_instance(outputdf.squeeze(), RF.predict_proba)
         st.write('LIME局部解释图')
         fig = exp.as_pyplot_figure()
         st.pyplot(fig, bbox_inches='tight')
         st.write(' ')
         st.write('上图显示了LIME的局部解释图，右侧的变量（绿色）表示对院内死亡的预测为正，左侧的变量（红色）表示对院内死亡的预测为负，下面的数值与变量的重要性相对应。')