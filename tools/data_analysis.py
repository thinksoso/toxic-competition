import matplotlib.pyplot as plt
#数据分析
#句子长度统计
sentences = list(set(df['less_toxic'].values.tolist()+df['more_toxic'].values.tolist()))
print(len(sentences))

sen_lengths = [len(i) for i in sentences]
plt.hist(sen_lengths,bins = 100)
plt.show()

sentences[-1]