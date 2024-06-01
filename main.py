import streamlit as st
from streamlit_navigation_bar import st_navbar

# region format
st.set_page_config(page_title="News Headlines Sarcasm Detection App", page_icon="🔗",
                   layout="wide")  # needs to be the first thing after the streamlit import

from io import BytesIO
from streamlit_echarts import st_echarts
from urllib.parse import urlparse
import chardet
import pandas as pd
from sentence_transformers import SentenceTransformer, util

import pickle
import torch
import numpy as np
from Headlines_RNN import MixtureOfExperts
from headline_data_set import HeadlineDataset
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import json
import keras_nlp
import keras
from keras.utils import custom_object_scope
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, GRU, LSTM, Bidirectional
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers import Activation, Dropout, Flatten
from keras.initializers import Constant
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
# from tensorflow.keras.models import load_model
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, word_tokenize

from sklearn import metrics
from sklearn.model_selection import train_test_split

from tqdm import tqdm
from subprocess import check_output
from collections import Counter
from wordcloud import WordCloud, ImageColorGenerator

import spacy
import string

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os, re, os, csv, math, codecs


finish = False
filter_h = [4,6,8]
device = torch.device("mps")
# beta_limit = 10000
# model_name = 'model1'

# @st.cache(allow_output_mutation=True)
@st.cache_data
def get_model1():
	## 加载模型
	checkpoint = torch.load("model_best.pth.tar")
	train_dataset = HeadlineDataset(
        csv_file='DATA/txt/headline_train.txt', 
        word_embedding_file='DATA/embeddings/headlines_filtered_embs.txt', 
        pad = max(filter_h) - 1,
        whole_data='DATA/txt/headlines_clean.txt',
    )
	parameters = {"filters": filter_h,
					"out_channels": 100,                  
                  	"max_length": train_dataset.max_l + 2  * (max(filter_h) - 1),
                  	"hidden_units": 64,
                  	"drop_prob": 0.2,
                  	"user_size": 400}
	model = MixtureOfExperts(parameters['filters'], parameters['out_channels'], parameters['max_length'], parameters['hidden_units'], 
                parameters['drop_prob'], 300, 256, 128, train_dataset.pretrained_embs)
	model = model.to(device)
	model.load_state_dict(checkpoint['state_dict'])
	model.eval()

	return train_dataset, model

@st.cache_data
def get_model2():

	model = keras.saving.load_model('bert.keras')

	return model

def get_model3():

	# 加载模型
	model = keras.saving.load_model('FastText_CNN.h5', custom_objects={'Sequential':Sequential})
	# adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	# model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
	return model

def preprocess_model3(text):
	MAX_NB_WORDS = 999000
	max_seq_len = 64
	tokenizer = RegexpTokenizer(r'\w+')
	stop_words = set(stopwords.words('english'))
	stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
	processed_docs = []
	for doc in tqdm(text):
		tokens = tokenizer.tokenize(doc)
		filtered = [word for word in tokens if word not in stop_words]
		processed_docs.append(" ".join(filtered))
	# end for

	print("tokenizing input data...")
	tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=False)
	tokenizer.fit_on_texts(processed_docs)
	word_seq = tokenizer.texts_to_sequences(processed_docs)
	word_index = tokenizer.word_index
	print("dictionary size: ", len(word_index))

	# pad sequences
	word_seq = sequence.pad_sequences(word_seq, maxlen=max_seq_len)

	return word_seq

def detect_text(text):

	if st.session_state.selected_model == "LSTM+CNN+Attention":
		train_dataset, model = get_model1()
		##数据预处理
		x = [0 for i in range(max(filter_h) - 1)]

		words = text.split()[:train_dataset.max_l]  # truncate words from test set
		for word in words:
			if word in train_dataset.word_idx:  # FIXME: skips unknown words
				x.append(train_dataset.word_idx[word])

		while len(x) < train_dataset.max_l + 2 * (max(filter_h) - 1):  # right padding
			x.append(0)
		x = torch.tensor(np.array(x).reshape(1, -1))

		input = torch.autograd.Variable(x, volatile=True).type(torch.LongTensor)
		input = input.to(device)

		##推理预测
		output = model(input)

		if output[0][0] > output[0][1]:
			return True
		else:
			return False

	elif st.session_state.selected_model == "BERT(slower)":
		model = get_model2()

		output = model.predict([text])

		if output[0][0] > output[0][1]:
			return True
		else:
			return False

	elif st.session_state.selected_model == "FastText+CNN":
		model = get_model3()

		output = model.predict(preprocess_model3([text]))

		if output[0] > 0.5:
			return True
		else:
			return False


def page_home():
	st.title('News')
	# 从JSON文件读取列表
	with open('news.json', 'r', encoding='utf-8') as f:
		news_list = json.load(f)

	#复选框选择是否过滤讽刺新闻
	filtered = st.checkbox('Filter out sarcastic headlines')

	# 复选框选择是否对每条新闻开启词云分析
	analysis = st.checkbox('Word cloud analysis')

	# 遍历新闻列表，为每个新闻创建一个expander
	for news in news_list:
		if filtered and detect_text(news['title']) == False:
			continue
		else:
			with st.expander(news['title']):  # 使用新闻标题作为expander的标题
				st.write(f"<p style='font-family: Arial, sans-serif; font-weight: bold; color: #ffa500; font-size: 35px'>{news['title']}</p>", unsafe_allow_html=True)
				st.write(news['content'])  # 展开后显示新闻内容
				st.write(f"<p style='font-family: Arial, sans-serif; font-weight: bold;'>{news['source']}</p>", unsafe_allow_html=True)
				st.write(news['time'])
				if analysis == True:
					words = news['content'].split()
					if st.session_state.stopwords_enabled == True:
						##去除停用词
						stop_words = set(stopwords.words('english'))
						filtered_words = [word for word in words if word.lower() not in stop_words]
					else:
						filtered_words = words
					word_freq = {}
					for word in filtered_words:
						if word in word_freq:
							word_freq[word] += 1
						else:
							word_freq[word] = 1
					wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

					# 创建一个matplotlib的Figure对象和一个Axes对象
					fig, ax = plt.subplots()

					# 使用matplotlib的imshow方法显示词云
					ax.imshow(wordcloud, interpolation='bilinear')
					ax.axis("off")  # 不显示坐标轴

					# 使用st.pyplot显示matplotlib的Figure
					st.pyplot(fig)


def page_about():
	st.title('About Our System')
	st.write('''  
	    Welcome to the About page of my system! Here you can find information about my product,  
	    mission, and how to contact me.  
	    ''')

	# 公司或项目的简短介绍
	st.subheader('Introduction')
	st.markdown('''  
	    This is a brief introduction to my system. We aim to provide users with a powerful and  
	    intuitive tool for sarcasm detection in news headlines.  
	    ''')

	# # 团队介绍
	# st.subheader('Our Team')
	# st.write('Our team consists of passionate and dedicated individuals who work hard to deliver the best.')
	# st.image('path_to_your_team_photo.jpg', caption='Our Awesome Team!')  # 替换为你的团队照片路径

	# 联系我们
	st.subheader('Contact Me')
	st.write('Feel free to contact me with any questions or suggestions you may have.')
	st.write('Email: [yangchengchang2008@163.com](mailto:yangchengchang2008@163.com)')
	# st.write('Website: [www.example.com](http://www.example.com)')


	# 版权信息
	st.subheader('Copyright')
	st.write('All content on this website is copyrighted and may not be reproduced without permission.')

def text_detection():
	st.title("News Headlines Sarcasm Detection")
	st.subheader("Please enter the news headline for sarcasm detection.")

	with st.form(key="my_form"):

		text = st.text_area(
			# Instructions
			"Enter the text:",
			# 'sample' variable that contains our keyphrases.
			value='',
			# The height
			height=200,
			# The tooltip displayed when the user hovers over the text area.
			help="Enter the text of news headline in the box below.",
			key="1",
		)

		submitted = st.form_submit_button(label="Submit")

		if submitted and text:
			if detect_text(text):
				st.success("No Sarcasm ✅")
			else:
				st.error("Sarcasm❗")
		elif submitted and not text:
			st.warning("❄️ Please input the headline.")




def batch_detection():
	st.title("News Headlines Sarcasm Detection")
	st.subheader("Please upload a tabular file containing news headlines data for sarcasm detection.")

	uploaded_file = st.file_uploader(
		"Upload a CSV file",
		help="""缺少样本？点击获取样本数据：https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection/data""")

	if uploaded_file is not None:

		try:

			result = chardet.detect(uploaded_file.getvalue())
			encoding_value = result["encoding"]

			if encoding_value == "UTF-16":
				white_space = True
			else:
				white_space = False

			df = pd.read_csv(
				uploaded_file,
				encoding=encoding_value,
				delim_whitespace=white_space,
				error_bad_lines=False,
			)
			# rename multi language columns
			df.rename(columns={"Adresse": "Address", "Dirección": "Address", "Indirizzo": "Address"}, inplace=True)
			number_of_rows = len(df)

			if number_of_rows > st.session_state.beta_limit:
				df = df[:st.session_state.beta_limit]
				st.caption("🚨 Imported rows over the beta limit, limiting to first " + str(st.session_state.beta_limit) + " rows.")

			if number_of_rows == 0:
				st.caption("Your sheet seems empty!")

			with st.expander("↕️ View raw data", expanded=False):
				st.write(df)

		except UnicodeDecodeError:
			st.warning(
				"""
                🚨 The file doesn't seem to load. Check the filetype, file format and Schema

                """
			)
			st.stop()

	else:
		st.stop()

	with st.form(key='filetype'):
		st.subheader("Please Select the Column to Match")
		kw_col = st.selectbox('Select the headline column:', df.columns)
		submitted = st.form_submit_button('Submit')

		if submitted:

			my_bar = st.progress(0)

			row_num = df.shape[0]

			word_freq1 = {}
			word_freq2 = {}

			nltk.download('stopwords')

			def predict(row):

				train_dataset = HeadlineDataset(
					csv_file='DATA/txt/headline_train.txt',
					word_embedding_file='DATA/embeddings/headlines_filtered_embs.txt',
					pad=max(filter_h) - 1,
					whole_data='DATA/txt/headlines_clean.txt',
				)

				words = row[kw_col].split()[:train_dataset.max_l]  # truncate words from test set

				if st.session_state.stopwords_enabled == True:
					##去除停用词
					stop_words = set(stopwords.words('english'))
					filtered_words = [word for word in words if word.lower() not in stop_words]
				else:
					filtered_words = words

				if detect_text(row[kw_col]):
					row["result"] = "✅No Sarcasm"
					for word in filtered_words:
						if word in word_freq1:
							word_freq1[word] += 1
						else:
							word_freq1[word] = 1
				else:
					row["result"] = "❌Sarcasm"
					for word in filtered_words:
						if word in word_freq2:
							word_freq2[word] += 1
						else:
							word_freq2[word] = 1

				my_bar.progress(float(row.name + 1) / row_num)

				return row[["result", kw_col]]

			result_df = df.apply(predict, axis=1)

			my_bar.empty()

			with st.expander("↕️ View result", expanded=True):
				st.write(result_df)

			wordcloud1 = WordCloud(background_color='white').generate_from_frequencies(word_freq1)
			wordcloud2 = WordCloud(background_color='white').generate_from_frequencies(word_freq2)
			fig, (ax1, ax2) = plt.subplots(1, 2)
			ax1.imshow(wordcloud1, interpolation="bilinear")
			ax1.axis("off")
			ax1.set_title("Wordcloud of non-sarcastic headlines.", fontsize=5)
			ax2.imshow(wordcloud2, interpolation="bilinear")
			ax2.axis("off")
			ax2.set_title("Wordcloud of sarcastic headlines.", fontsize=5)
			st.pyplot(fig)

def page_setting():

	st.title("Settings")

	options = ('LSTM+CNN+Attention', 'BERT(slower)', 'FastText+CNN')

	# 找到默认值在选项列表中的索引
	# default_index = options.index(st.session_state.selected_model)

	selected_model = st.selectbox('Select Model', options, key='model_select')

	# 如果用户做了选择，更新 session_state 中的值
	if selected_model is not None:
		st.session_state.selected_model = selected_model

	st.write('The following shows the perfomance of the three models embedded in this system.')
	st.image("/Users/yangchengchang/Documents/Graduation Project/streamlit/performance.png", width=600)

	# 使用 slider 获取用户的选择，并尝试使用 session_state 中的值作为默认值
	slider_min = 0
	slider_max = 10000
	beta_limit = st.slider('Set the Limitation of the Data', min_value=slider_min, max_value=slider_max,
							 key='slider_key', help="""Set the maximum number of rows to read from the CSV file.""")

	# 如果用户移动了滑块，更新 session_state 中的值
	if beta_limit is not None:
		st.session_state.beta_limit = beta_limit

	# 创建单选框让用户选择是否开启停用词过滤
	stopwords_options = ['Off', 'On']
	stopwords_choice = st.radio('Stop word filtering', stopwords_options, key='stopwords_choice',
								help="""Whether to remove stop words during word cloud analysis""")

	# 更新 session_state 中的变量以反映用户的选择
	st.session_state.stopwords_enabled = stopwords_choice == 'On'


# beta_limit = st.sidebar.slider("Set the Limitation of the Data", value=10000, help="""Set the maximum number of rows to read from the CSV file.""")

# st.title("News Headlines Sarcasm Detection")
# model_name = st.sidebar.selectbox('Select Model', ('model1', 'model2'))
# beta_limit = st.sidebar.slider("Set the Limitation of the Data", value=10000, help="""Set the maximum number of rows to read from the CSV file.""")

def page_feedback():
	# 标题
	st.title('User Feedback')

	# 用户输入误判的新闻标题
	misclassified_title = st.text_input('Enter the misclassified news headline:', '')

	# 用户选择该标题是否具有讽刺意味
	options = ['Yes', 'No']  # 假设你使用中文界面，所以这里用了中文选项
	is_sarcastic_index = st.selectbox('Is the headline sarcastic?', options, index=1)  # index=1 默认选中“否”

	# 如果需要转换为布尔值，可以这样做：
	is_sarcastic = is_sarcastic_index == 0  # 假设 '是' 是第一个选项（索引为0）

	# 假设我们有一个变量来存储历史反馈（实际应用中可能需要保存到数据库）
	feedback_list = []

	# 加载已有的反馈（如果文件存在）
	try:
		with open('feedbacks.json', 'r') as file:
			feedback_list = json.load(file)
	except FileNotFoundError:
		pass  # 如果文件不存在，则忽略

	# 提交按钮
	if st.button('Submit'):
		# 收集用户反馈
		feedback = {
			'title': misclassified_title,
			'is_sarcastic': is_sarcastic
		}

		# 添加到反馈列表（实际应用中应保存到数据库）
		append
		# 将反馈列表保存到JSON文件
		with open('feedbacks.json', 'w') as file:
			json.dump(feedback_list, file, indent=4)  # indent使输出更易读

		st.success('Thank you for your feedback!😊')

		# 可以选择清空输入框以便下一次输入
		misclassified_title = ''
		is_sarcastic = False

	if feedback_list:
		st.subheader('Your Submitted Feedbacks')

		# 创建一个列表，包含所有反馈的相关信息
		data = [{"Headline": fb['title'],
				 "Is Sarcastic": "Yes" if fb['is_sarcastic'] else "No"}
				for idx, fb in enumerate(feedback_list)]

		# 使用st.table显示反馈列表
		st.table(data)
	else:
		st.write("No feedbacks submitted yet.")


def main():

	# 设置初始页面为Home
	session_state = st.session_state

	# 初始化 session_state 中的变量，如果还没有的话
	# if 'selected_model' not in st.session_state:
	# 	st.session_state.selected_model = "HNN"  # 设置默认值

	# if 'beta_limit' not in st.session_state:
	# 	st.session_state.beta_limit = 5000

	# 初始化 session_state 中的变量，如果还没有的话
	# if 'stopwords_enabled' not in st.session_state:
	# 	st.session_state.stopwords_enabled = True  # 默认启用停用词过滤

	if 'page' not in session_state:
		session_state['page'] = 'Home'

	# st.sidebar.title("🧭Navigate")
	# 导航栏
	# page = st.sidebar.radio('', ['🏠Home', '🔼️Sarcasm Detection (Text)', '⏫Batch Sarcasm Detection (File)', '🔧Setting', '📥Feedback',  '❓️About'])
	# page = st_navbar(['🏠Home', '🔼️Sarcasm Detection (Text)', '⏫Batch Sarcasm Detection (File)', '🔧Setting', '📥Feedback', '❓️About'])
	page = st_navbar(
		['🏠Home', '📃Text Detection', '📑File Detection', '🔧Setting', '📥Feedback',
		 '❓️About'])
	if page == '🏠Home':
		page_home()
	elif page == '❓️About':
		page_about()
	elif page == '📃Text Detection':
		text_detection()
	elif page == '📑File Detection':
		batch_detection()
	elif page == '📥Feedback':
		page_feedback()
	elif page == '🔧Setting':
		page_setting()


if __name__ == '__main__':
	main()


# st.write(
#     "[![this is an image link](https://i.imgur.com/Ex8eeC2.png)](https://www.patreon.com/leefootseo) [Become a Patreon for Early Access, Support & More!](https://www.patreon.com/leefootseo)  |  Made in [![this is an image link](https://i.imgur.com/iIOA6kU.png)](https://www.streamlit.io/) by [@LeeFootSEO](https://twitter.com/LeeFootSEO)")
# accuracy_slide = st.sidebar.slider("Set Cluster Accuracy: 0-100", value=75)
# min_cluster_size = st.sidebar.slider("Set Minimum Cluster Size: 0-100", value=2)
# source_filter = st.sidebar.text_input('Filter Source URL Type')
# destination_filter = st.sidebar.text_input('Filter Destination URL Type')
# min_similarity = accuracy_slide / 100


# TextTab, FileTab = st.tabs(["Text", "File"])
#
# with TextTab:
# 	st.subheader("Please enter the news headline for sarcasm detection.")
#
# 	with st.form(key="my_form"):
#
# 		text = st.text_area(
#             # Instructions
#             "Enter the text:",
#             # 'sample' variable that contains our keyphrases.
#             value='',
#             # The height
#             height=200,
#             # The tooltip displayed when the user hovers over the text area.
#             help="Enter the text of news headline in the box below.",
#             key="1",
#         )
#
# 		submitted = st.form_submit_button(label="Submit")
# 		if submitted and text:
#
# 			##数据预处理
# 			x = [0 for i in range(max(filter_h) - 1)]
#
# 			words = text.split()[:train_dataset.max_l] # truncate words from test set
# 			for word in words:
# 				if word in train_dataset.word_idx: # FIXME: skips unknown words
# 					x.append(train_dataset.word_idx[word])
#
# 			while len(x) < train_dataset.max_l + 2 * (max(filter_h) - 1) : # right padding
# 				x.append(0)
# 			x = torch.tensor(np.array(x).reshape(1, -1))
#
# 			input = torch.autograd.Variable(x, volatile=True).type(torch.LongTensor)
# 			input = input.to(device)
#
# 			##推理预测
# 			output = model(input)
#
# 			if output[0][0] > output[0][1]:
# 				st.success("No Sarcasm ✅")
# 			else:
# 				st.error("Sarcasm❗")
#
# 		elif submitted and not text:
# 			st.warning("❄️ Please input the headline.")
#
#
# with FileTab:
# 	st.subheader("Please upload a tabular file containing news headlines data for sarcasm detection.")
#
# 	uploaded_file = st.file_uploader(
# 	    "Upload a CSV file",
# 	    help="""缺少样本？点击获取样本数据：https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection/data""")
#
# 	if uploaded_file is not None:
#
# 	    try:
#
# 	        result = chardet.detect(uploaded_file.getvalue())
# 	        encoding_value = result["encoding"]
#
# 	        if encoding_value == "UTF-16":
# 	            white_space = True
# 	        else:
# 	            white_space = False
#
# 	        df = pd.read_csv(
# 	            uploaded_file,
# 	            encoding=encoding_value,
# 	            delim_whitespace=white_space,
# 	            error_bad_lines=False,
# 	        )
# 	        # rename multi language columns
# 	        df.rename(columns={"Adresse": "Address", "Dirección": "Address", "Indirizzo": "Address"}, inplace=True)
# 	        number_of_rows = len(df)
#
# 	        if number_of_rows > beta_limit:
# 	            df = df[:beta_limit]
# 	            st.caption("🚨 Imported rows over the beta limit, limiting to first " + str(beta_limit) + " rows.")
#
# 	        if number_of_rows == 0:
# 	            st.caption("Your sheet seems empty!")
#
# 	        with st.expander("↕️ View raw data", expanded=False):
# 	            st.write(df)
#
# 	    except UnicodeDecodeError:
# 	        st.warning(
# 	            """
# 	            🚨 The file doesn't seem to load. Check the filetype, file format and Schema
#
# 	            """
# 	        )
#
# 	else:
# 	    st.stop()
#
#
#
#
# 	with st.form(key='filetype'):
# 	    st.subheader("Please Select the Column to Match")
# 	    kw_col = st.selectbox('Select the headline column:', df.columns)
# 	    submitted = st.form_submit_button('Submit')
#
# 	    if submitted:
#
# 	    	my_bar = st.progress(0)
#
# 	    	row_num = df.shape[0]
#
# 	    	word_freq1 = {}
# 	    	word_freq2 = {}
#
# 	    	nltk.download('stopwords')
#
# 	    	def predict(row):
#
# 	    		##数据预处理
# 	    		x = [0 for i in range(max(filter_h) - 1)]
# 	    		words = row[kw_col].split()[:train_dataset.max_l] # truncate words from test set
#
# 	    		for word in words:
# 	    			if word in train_dataset.word_idx: # FIXME: skips unknown words
# 	    				x.append(train_dataset.word_idx[word])
#
# 	    		while len(x) < train_dataset.max_l + 2 * (max(filter_h) - 1): # right padding
# 	    			x.append(0)
#
# 	    		x = torch.tensor(np.array(x).reshape(1, -1))
#
# 	    		input = torch.autograd.Variable(x, volatile=True).type(torch.LongTensor)
# 	    		input = input.to(device)
#
# 	    		##推理预测
# 	    		with torch.no_grad():
# 	    			output = model(input)
#
# 	    		##去除停用词
# 	    		stop_words = set(stopwords.words('english'))
#
# 	    		filtered_words = [word for word in words if word.lower() not in stop_words]
#
# 	    		if output[0][0] > output[0][1]:
# 	    			row["result"] = "✅No Sarcasm"
# 	    			for word in filtered_words:
# 	    				if word in word_freq1:
# 	    					word_freq1[word] += 1
# 	    				else:
# 	    					word_freq1[word] = 1
# 	    		else:
# 	    			row["result"] = "❌Sarcasm"
# 	    			for word in filtered_words:
# 	    				if word in word_freq2:
# 	    					word_freq2[word] += 1
# 	    				else:
# 	    					word_freq2[word] = 1
#
# 	    		my_bar.progress(float(row.name + 1) / row_num)
#
# 	    		return row[["result", kw_col]]
#
# 	    	result_df = df.apply(predict, axis=1)
#
# 	    	my_bar.empty()
#
# 	    	with st.expander("↕️ View result", expanded=True):
# 	    		st.write(result_df)
#
# 	    	wordcloud1 = WordCloud().generate_from_frequencies(word_freq1)
# 	    	wordcloud2 = WordCloud().generate_from_frequencies(word_freq2)
# 	    	fig,(ax1, ax2) = plt.subplots(1, 2)
# 	    	ax1.imshow(wordcloud1, interpolation="bilinear")
# 	    	ax1.axis("off")
# 	    	ax1.set_title("Wordcloud of non-sarcastic headlines.", fontsize=5)
# 	    	ax2.imshow(wordcloud2, interpolation="bilinear")
# 	    	ax2.axis("off")
# 	    	ax2.set_title("Wordcloud of sarcastic headlines.", fontsize=5)
# 	    	st.pyplot(fig)
