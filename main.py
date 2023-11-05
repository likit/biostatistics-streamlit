import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from PIL import Image

scores = pd.read_csv('datasets/exam1.csv', header=None)

st.title('Welcome to Biostatistics Learning Platform!')
st.markdown('Developed by ***Likit Preeyanon, Ph.D.*** E-mail: *likit.pre@mahidol.edu*')

st.header('Central Limit Theorem!')
st.markdown('สมมติว่าเราทำการเก็บรวบรวมคะแนนสอบกลางภาคของนักศึกษารายวิชาปรสิตวิทยาทางการแพทย์ จำนวน 80 คน ได้ข้อมูลต่อไปนี้')
st.write(scores)

mean_scores = scores.mean()[0]

st.markdown('สถิติทั่วไปมีค่าดังนี้')
st.write(scores.describe())

st.markdown('เมื่อนำมาสร้างกราฟ histogram พบว่ามีการกระจายของข้อมูลดังรูป')

fig, ax = plt.subplots()
ax.hist(scores, bins=20)

st.pyplot(fig)

st.markdown(r'หากนักศึกษาทำการสุ่มข้อมูลจากนักศึกษาครั้งละ $n$ คนและนำมาหาค่าเฉลี่ย $(sample\ mean)$ โดยทำซ้ำจำนวน $n$ ครั้ง และนำมาสร้างกราฟ histogram ของค่าเฉลี่ย จะได้กราฟแสดงการกระจายตัวของ sample means เป็นอย่างไร')


def plot_deviation_sample_mean_chart(scores_mean, num_sample, num_trials):
    deviations = []
    sample_means = []

    for i in range(num_trials):
        sample_mean = scores.sample(num_sample).mean()[0]
        deviations.append(scores_mean - sample_mean)
        sample_means.append(sample_mean)

    fig, axes = plt.subplots(2)
    axes[0].hlines(y=0, xmin=0, xmax=len(deviations), color='red')
    axes[0].set_ylim(-5, 5)
    std = np.std(sample_means)
    hvalues, bins, _ = axes[1].hist(sample_means, bins=10)
    axes[0].plot(deviations)
    axes[1].vlines(x=scores_mean, ymax=max(hvalues), ymin=0, color='red', linestyles='--')
    axes[1].vlines(x=scores_mean - std, ymax=max(hvalues), ymin=0, color='red', linestyles='--')
    axes[1].vlines(x=scores_mean + std, ymax=max(hvalues), ymin=0, color='red', linestyles='--')
    axes[1].vlines(x=scores_mean - (2 * std), ymax=max(hvalues), ymin=0, color='red', linestyles='--')
    axes[1].vlines(x=scores_mean + (2 * std), ymax=max(hvalues), ymin=0, color='red', linestyles='--')
    axes[1].vlines(x=scores_mean - (3 * std), ymax=max(hvalues), ymin=0, color='red', linestyles='--')
    axes[1].vlines(x=scores_mean + (3 * std), ymax=max(hvalues), ymin=0, color='red', linestyles='--')
    axes[1].annotate('Mean', (scores_mean, 0))
    axes[1].annotate('+1SD', (scores_mean + std, 0))
    axes[1].annotate('-1SD', (scores_mean - std, 0))
    axes[1].annotate('+2SD', (scores_mean + (2 * std), 0))
    axes[1].annotate('-2SD', (scores_mean - (2 * std), 0))
    axes[1].annotate('+3SD', (scores_mean + (3 * std), 0))
    axes[1].annotate('-3SD', (scores_mean - (3 * std), 0))
    axes[1].hlines(y=max(hvalues) / 3, xmin=(scores_mean - (1 * std)), xmax=(scores_mean + (1 * std)), color='black',
                   linestyles='-')
    axes[1].annotate('68.3%', (scores_mean, max(hvalues) / 3))
    axes[1].hlines(y=max(hvalues) / 2, xmin=(scores_mean - (2 * std)), xmax=(scores_mean + (2 * std)), color='black',
                   linestyles='-')
    axes[1].annotate('95.5%', (scores_mean, max(hvalues) / 2))

    st.pyplot(fig)

    st.markdown(r'Standard deviation of the sample means is {}'.format(np.mean([d**2 for d in deviations])))
    st.markdown(r'Standard Error of the Mean $(SEM)$ is {}'.format(np.std(scores)[0] / np.sqrt(num_sample)))


num_sample = st.slider('Number of students', min_value=10, max_value=70, step=10)
num_trials = st.slider('Number of trials', min_value=50, max_value=1000, step=50)

plot_deviation_sample_mean_chart(scores_mean=mean_scores, num_sample=num_sample, num_trials=num_trials)


def plot_se(sigma, n=[10, 20, 40, 80, 160, 320, 640, 1280]):
    ses = sigma / np.sqrt(n)
    st.dataframe({'SE': ses, 'n': n})


sigma = st.slider('Sigma', min_value=50, max_value=100, step=10)

plot_se(sigma)

st.header('Z-statistic or Z-score')
st.markdown(r'$Z-statistic$ เหมือนกับ $Z-score$')
st.markdown(r'***$Z-statistic$*** คำนวณจากสมการ $Z = \frac{\overline{x}-\mu}{\frac{\sigma}{\sqrt{n}}}$')
st.markdown(
    r'ค่า ***$Z score$*** มีค่าเท่ากับ $Z statistic$ เมื่อ $n=1$.')
st.markdown(
    r'สมมติเราสุ่มตัวอย่างจากประชากรที่มีค่า $\mu=50$ และ $\sigma=10$ มาสามกลุ่ม โดยทั้งสามกลุ่มมีค่า ${\overline{x}}=52$ และจำนวนตัวอย่างคือ $30, 60, 100$ ตามลำดับ จงคำนวณค่า $Z statistic$')


def calculate_z_stat(mean, sigma, n, sample_mean):
    return (sample_mean - mean) / (sigma / np.sqrt(n))


def plot_norm_pdf(pop_mean, sigma, n):
    x_axis = np.linspace(0, 100, int(n))
    fig, ax = plt.subplots()
    ax.plot(x_axis, norm.pdf(x_axis, pop_mean, sigma))
    st.pyplot(fig)


def plot_z_score(sample_mean, mean, sigma, n):
    zscore = (sample_mean - mean) / (sigma / np.sqrt(n))
    fig, ax = plt.subplots()
    x_axis = np.linspace(-3, 3, 100)
    ax.plot(x_axis, norm.pdf(x_axis, 0, 1))
    ax.vlines(x=zscore, ymin=0, ymax=0.4, color='red')
    ax.annotate(str(f'{zscore:.3f}'), (zscore, 0.01))
    st.pyplot(fig)


pop_mean = st.number_input('Population mean')
sigma = st.number_input('Population standard deviation')
n = st.number_input('Sample size')
sample_mean = st.number_input('Sample mean')
zscore = (sample_mean - pop_mean) / (sigma / np.sqrt(n))

st.markdown('Z statistic = {}'.format(calculate_z_stat(pop_mean, sigma, n, sample_mean)))
st.markdown('SEM = {}'.format(sigma / np.sqrt(n)))
#plot_norm_pdf(pop_mean, sigma, n)
plot_z_score(sample_mean, pop_mean, sigma, n)

st.markdown('เราสามารถคำนวณหาค่าความน่าจะเป็นของค่า Z statistic ในโปรแกรม Excel ได้จากสูตร')
st.code(f'NORM.S.DIST({zscore:.2f}, TRUE) = {norm.cdf(zscore):.2f}')
st.markdown(f'หมายความว่าโอกาสที่จะได้ค่าต่ำกว่าหรือเท่ากับ {sample_mean} ({zscore:.2f}SD) คือ {norm.cdf(zscore)*100:.2f}%')
st.markdown(f'หมายความว่าโอกาสที่จะได้ค่ามากกว่า {sample_mean} ({zscore:.2f}SD) คือ {(1 - norm.cdf(zscore))*100:.2f}%')

st.markdown('ในทางกลับกันเราสามารถคำนวณหา Z statistic จากความน่าจะเป็นได้ในโปรแกรม Excel ได้จากสูตร')
st.code(f'NORM.S.INV({norm.cdf(zscore):.2f}) = {norm.ppf(norm.cdf(zscore)):.2f}')

normal_dist = Image.open('images/norm_dist.jpeg')

st.image(normal_dist, caption='Normal Distribution')

st.header('Self Check')
st.markdown(r'1) ผลการสอบภาคปฏิบัติรายวิชาปรสิตวิทยาได้คะแนนเฉลี่ยเท่ากับ $24$ คะแนน โดยมีค่า $SD=4.6$ จงหาความน่าจะเป็นที่นักศึกษาจะได้คะแนนน้อยกว่า $18$ คะแนน')
with st.expander('See an answer'):
    st.markdown(r'เราสามารถคำนวณหาค่าความน่าจะเป็นของค่า $Z-statistic$ ในโปรแกรม Excel ได้จากสูตร')
    zscore = (18 - 24) / (4.6 / np.sqrt(1))
    st.markdown(r'$Z-statistic = \frac{18-24}{\frac{4.6}{\sqrt{1}}} = -1.3043$')
    st.code(f'NORM.S.DIST({zscore:.4f}, TRUE) = {norm.cdf(zscore):.4f}')


st.markdown(r'2) ครูใหญ่โรงเรียนแห่งหนึ่งได้รับผลการทดสอบ IQ จากนักเรียนคนหนึ่ง หากครูทราบว่าผลเฉลี่ย IQ ของนักเรียนในวัยเดียวกันทั้งประเทศคือ $100$ และค่า $SD$ คือ $15$ ค่าทางสถิติใดที่จะช่วยแปลผลคะแนนของนักเรียนคนนี้หากการกระจายตัวของคะแนน IQ เป็นแบบ normal distribution (Z distribution)')
with st.expander('See an answer'):
    st.markdown(r'เราสามารถคำนวณหาค่าความน่าจะเป็นของค่า $Z-statistic$ ในโปรแกรม Excel ได้จากสูตร')
    zscore = (80 - 100) / (15 / np.sqrt(1))
    st.markdown(r'$Z-statistic = \frac{80-100}{\frac{15}{\sqrt{1}}} = -1.3333$')
    st.code(f'NORM.S.DIST({zscore:.4f}, TRUE) = {norm.cdf(zscore):.4f}')


st.markdown(r'3.1) นักเทคนิคการแพทย์ต้องการศึกษาผลของการทานอาหารมังสะวิรัตกับค่าคลอเลสเตอรอลในเลือด หากว่าค่าคลอเลสเตอรอลในผู้ชายระหว่างอายุ $20-65$ ปี มีการกระจายตัวแบบ normal distribution ด้วยค่าเฉลี่ยเท่ากับ $210 mg/dL$ และค่า $SD = 45$ mg/dL หากนักเทคนิคการแพทย์ได้ทำการเก็บตัวอย่างจำนวน $40$ คนที่อยู่ในกลุ่มอายุดังกล่าวหลังจากได้ทำการติดตามการทานอาหารมาเป็นเวลาหนึ่งปีและพบว่าค่าเฉลี่ยในกลุ่มนี้คือ $190 mg/dL$ ค่าสถิติใดที่ควรใช้ในการแปลผลนี้')
with st.expander('See an answer'):
    st.markdown(r'เราสามารถคำนวณหาค่าความน่าจะเป็นของค่า $Z-statistic$ ในโปรแกรม Excel ได้จากสูตร')
    zscore = (190 - 210) / (45 / np.sqrt(40))
    st.markdown(r'$Z-statistic = \frac{190-210}{\frac{45}{\sqrt{40}}} = -2.8109$')
    st.code(f'NORM.S.DIST({zscore:.4f}, TRUE) = {norm.cdf(zscore):.4f}')


st.markdown(r'3.2) หากกำหนดค่า $p-value = 0.05$ ผลการทานอาหารมังสะวิรัตแตกต่างจากค่าเฉลี่ยทั่วไปอย่างมีนัยสำคัญหรือไม่')
with st.expander('See an answer'):
    st.markdown(r'เนื่องจากค่าความน่าจะเป็นที่ได้ค่า $\overline{x} = 190mg/dL$ จากกลุ่มประชากรที่มีค่า $\mu=210 mg/dL$ มีค่า $0.0025$ ซึ่งน้อยกว่าค่า $\alpha=(0.05)$ จึงถือว่ามีความแตกต่างอย่างมีนัยสำคัญ')

st.markdown(r'3.3) หากค่าเฉลี่ยจากกลุ่มทดลองมีค่าเท่ากับ $220 mg/dL$ จะมีความแตกต่างจากค่าเฉลี่ยอย่างมีนัยสำคัญหรือไม่')
with st.expander('See an answer'):
    st.markdown(r'เราสามารถคำนวณหาค่าความน่าจะเป็นของค่า $Z-statistic$ ในโปรแกรม Excel ได้จากสูตร')
    zscore = (220 - 210) / (45 / np.sqrt(40))
    st.markdown(r'$Z-statistic = \frac{190-210}{\frac{45}{\sqrt{40}}} = 1.4055$')
    st.code(f'NORM.S.DIST({zscore:.4f}, TRUE) = {norm.cdf(zscore):.4f}')

    st.markdown(r'ดังนั้นค่าความน่าจะเป็นที่จะได้ค่าเฉลี่ยมากกว่า $220 mg/dL$ คือ $1-{:.4f}={:.4f}$ ซึ่งเกินกว่าค่า $\alpha$ จึงถือว่าไม่แตกต่างอย่างมีนัยสำคัญ'.format(norm.cdf(zscore), 1-norm.cdf(zscore)))

