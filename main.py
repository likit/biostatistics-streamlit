import random

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
st.markdown(
    'สมมติว่าเราทำการเก็บรวบรวมคะแนนสอบกลางภาคของนักศึกษารายวิชาปรสิตวิทยาทางการแพทย์ จำนวน 80 คน ได้ข้อมูลต่อไปนี้')
st.write(scores)

mean_scores = scores.mean()[0]

st.markdown('สถิติทั่วไปมีค่าดังนี้')
st.write(scores.describe())

st.markdown('เมื่อนำมาสร้างกราฟ histogram พบว่ามีการกระจายของข้อมูลดังรูป')

fig, ax = plt.subplots()
ax.hist(scores, bins=10)

st.pyplot(fig)

st.markdown(
    r'หากนักศึกษาทำการสุ่มข้อมูลจากนักศึกษาครั้งละ $n$ คนและนำมาหาค่าเฉลี่ย $(sample\ mean)$ โดยทำซ้ำจำนวน $n$ ครั้ง และนำมาสร้างกราฟ histogram ของค่าเฉลี่ย จะได้กราฟแสดงการกระจายตัวของ sample means เป็นอย่างไร')


def plot_deviation_sample_mean_chart(scores_mean, num_sample, num_trials):
    deviations = []
    sample_means = []

    for i in range(num_trials):
        sample_mean = scores.sample(num_sample).mean()[0]
        deviations.append(scores_mean - sample_mean)
        sample_means.append(sample_mean)

    fig, axes = plt.subplots(2)
    axes[0].set_ylim(-5, 5)
    std = np.std(sample_means)
    hvalues, bins, _ = axes[1].hist(sample_means, bins=10)
    axes[0].plot(deviations)
    axes[0].hlines(y=0, xmin=0, xmax=len(deviations), color='red')
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

    # st.markdown(r'Standard deviation of the sample means is {}'.format(np.mean([d ** 2 for d in deviations])))
    st.markdown(r'$Standard\ Error\ of\ the\ Mean\ (SEM)={:.2f}$'.format(np.std(scores)[0] / np.sqrt(num_sample)))


num_sample = st.slider('Number of students', min_value=10, max_value=70, step=10)
num_trials = st.slider('Number of trials', min_value=50, max_value=1000, step=50)

plot_deviation_sample_mean_chart(scores_mean=mean_scores, num_sample=num_sample, num_trials=num_trials)

st.header('Note')
st.markdown(r'เมื่อจำนวน $n$ หรือ $sample\ size$ เพิ่มขึ้น ค่า $SEM$ ลดลง')
st.markdown(r'เมื่อจำนวน $trials$ เพิ่มขึ้น กราฟจะเป็นลักษณะ normal distribution มากขึ้น ทั้งนี้ ไม่ขึ้นกับลักษณะการกระจายตัวของข้อมูลดั้งเดิม')
st.markdown(r'ค่า $SE$ คำนวณจาก $\frac{\sigma}{\sqrt(n)}$')

def plot_se(sigma, n=[1, 10, 20, 40, 80, 160, 320, 640, 1280]):
    ses = sigma / np.sqrt(n)
    st.dataframe({'SE': ses, 'n': n})
    fig, ax = plt.subplots()
    ax.plot(n, ses, '-*')
    ax.set_ylabel('SE')
    ax.set_xlabel('sample size')
    ax.set_title('Sample size vs SE')
    st.pyplot(fig)


st.header('Z-statistic or Z-score')
st.markdown(r'$Z-statistic$ เหมือนกับ $Z-score$')
st.markdown(r'***$Z-statistic$*** คำนวณจากสมการ $Z = \frac{\overline{x}-\mu}{\frac{\sigma}{\sqrt{n}}}$')
sigma = st.slider('Sigma', min_value=50, max_value=100, step=10)

plot_se(sigma)

st.markdown(
    r'ค่า ***$Z-score$*** มีค่าเท่ากับ $Z-statistic$ เมื่อ $n=1$.')
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


st.header('Z-statistic Calculator')
pop_mean = st.number_input(r'$\mu$')
sigma = st.number_input(r'$\sigma$')
n = st.number_input(r'$n$')
sample_mean = st.number_input(r'$\overline{x}$', key='sample_mean')
zscore = (sample_mean - pop_mean) / (sigma / np.sqrt(n))

st.markdown('Z statistic = {}'.format(calculate_z_stat(pop_mean, sigma, n, sample_mean)))
st.markdown('SEM = {}'.format(sigma / np.sqrt(n)))
# plot_norm_pdf(pop_mean, sigma, n)
plot_z_score(sample_mean, pop_mean, sigma, n)

st.markdown('เราสามารถคำนวณหาค่าความน่าจะเป็นของค่า Z statistic ในโปรแกรม Excel ได้จากสูตร')
st.code(f'NORM.S.DIST({zscore:.2f}, TRUE) = {norm.cdf(zscore):.2f}')
st.markdown(
    f'หมายความว่าโอกาสที่จะได้ค่าต่ำกว่าหรือเท่ากับ {sample_mean} ({zscore:.2f}SD) คือ {norm.cdf(zscore) * 100:.2f}%')
st.markdown(
    f'หมายความว่าโอกาสที่จะได้ค่ามากกว่า {sample_mean} ({zscore:.2f}SD) คือ {(1 - norm.cdf(zscore)) * 100:.2f}%')

st.markdown('ในทางกลับกันเราสามารถคำนวณหา Z statistic จากความน่าจะเป็นได้ในโปรแกรม Excel ได้จากสูตร')
st.code(f'NORM.S.INV({norm.cdf(zscore):.2f}) = {norm.ppf(norm.cdf(zscore)):.2f}')

normal_dist = Image.open('images/norm_dist.jpeg')

st.image(normal_dist, caption='Normal Distribution')

st.header('Self Check')
st.markdown(
    r'1) ผลการสอบภาคปฏิบัติรายวิชาปรสิตวิทยาได้คะแนนเฉลี่ยเท่ากับ $24$ คะแนน โดยมีค่า $SD=4.6$ จงหาความน่าจะเป็นที่นักศึกษาจะได้คะแนนน้อยกว่า $18$ คะแนน')
with st.expander('See an answer'):
    st.markdown(r'เราสามารถคำนวณหาค่าความน่าจะเป็นของค่า $Z-statistic$ ในโปรแกรม Excel ได้จากสูตร')
    zscore = (18 - 24) / (4.6 / np.sqrt(1))
    st.markdown(r'$Z-statistic = \frac{18-24}{\frac{4.6}{\sqrt{1}}} = -1.3043$')
    st.code(f'NORM.S.DIST({zscore:.4f}, TRUE) = {norm.cdf(zscore):.4f}')

st.markdown(
    r'2) ครูใหญ่โรงเรียนแห่งหนึ่งได้รับผลการทดสอบ IQ จากนักเรียนคนหนึ่ง หากครูทราบว่าผลเฉลี่ย IQ ของนักเรียนในวัยเดียวกันทั้งประเทศคือ $100$ และค่า $SD$ คือ $15$ ค่าทางสถิติใดที่จะช่วยแปลผลคะแนนของนักเรียนคนนี้หากการกระจายตัวของคะแนน IQ เป็นแบบ normal distribution (Z distribution)')
with st.expander('See an answer'):
    st.markdown(r'เราสามารถคำนวณหาค่าความน่าจะเป็นของค่า $Z-statistic$ ในโปรแกรม Excel ได้จากสูตร')
    zscore = (80 - 100) / (15 / np.sqrt(1))
    st.markdown(r'$Z-statistic = \frac{80-100}{\frac{15}{\sqrt{1}}} = -1.3333$')
    st.code(f'NORM.S.DIST({zscore:.4f}, TRUE) = {norm.cdf(zscore):.4f}')

st.markdown(
    r'3.1) นักเทคนิคการแพทย์ต้องการศึกษาผลของการทานอาหารมังสะวิรัตกับค่าคลอเลสเตอรอลในเลือด หากว่าค่าคลอเลสเตอรอลในผู้ชายระหว่างอายุ $20-65$ ปี มีการกระจายตัวแบบ normal distribution ด้วยค่าเฉลี่ยเท่ากับ $210 mg/dL$ และค่า $SD = 45$ mg/dL หากนักเทคนิคการแพทย์ได้ทำการเก็บตัวอย่างจำนวน $40$ คนที่อยู่ในกลุ่มอายุดังกล่าวหลังจากได้ทำการติดตามการทานอาหารมาเป็นเวลาหนึ่งปีและพบว่าค่าเฉลี่ยในกลุ่มนี้คือ $190 mg/dL$ ค่าสถิติใดที่ควรใช้ในการแปลผลนี้')
with st.expander('See an answer'):
    st.markdown(r'เราสามารถคำนวณหาค่าความน่าจะเป็นของค่า $Z-statistic$ ในโปรแกรม Excel ได้จากสูตร')
    zscore = (190 - 210) / (45 / np.sqrt(40))
    st.markdown(r'$Z-statistic = \frac{190-210}{\frac{45}{\sqrt{40}}} = -2.8109$')
    st.code(f'NORM.S.DIST({zscore:.4f}, TRUE) = {norm.cdf(zscore):.4f}')

st.markdown(r'3.2) หากกำหนดค่า $p-value = 0.05$ ผลการทานอาหารมังสะวิรัตแตกต่างจากค่าเฉลี่ยทั่วไปอย่างมีนัยสำคัญหรือไม่')
with st.expander('See an answer'):
    st.markdown(
        r'เนื่องจากค่าความน่าจะเป็นที่ได้ค่า $\overline{x} = 190mg/dL$ จากกลุ่มประชากรที่มีค่า $\mu=210 mg/dL$ มีค่า $0.0025$ ซึ่งน้อยกว่าค่า $\alpha=(0.05)$ จึงถือว่ามีความแตกต่างอย่างมีนัยสำคัญ')

st.markdown(r'3.3) หากค่าเฉลี่ยจากกลุ่มทดลองมีค่าเท่ากับ $220 mg/dL$ จะมีความแตกต่างจากค่าเฉลี่ยอย่างมีนัยสำคัญหรือไม่')
with st.expander('See an answer'):
    st.markdown(r'เราสามารถคำนวณหาค่าความน่าจะเป็นของค่า $Z-statistic$ ในโปรแกรม Excel ได้จากสูตร')
    zscore = (220 - 210) / (45 / np.sqrt(40))
    st.markdown(r'$Z-statistic = \frac{190-210}{\frac{45}{\sqrt{40}}} = 1.4055$')
    st.code(f'NORM.S.DIST({zscore:.4f}, TRUE) = {norm.cdf(zscore):.4f}')

    st.markdown(r'ดังนั้นค่าความน่าจะเป็นที่จะได้ค่าเฉลี่ยมากกว่า $220 mg/dL$ คือ $1-{:.4f}={:.4f}$ ซึ่งเกินกว่าค่า '
                r'$\alpha$ จึงถือว่าไม่แตกต่างอย่างมีนัยสำคัญ'.format(norm.cdf(zscore), 1 - norm.cdf(zscore)))

st.header('Confidence Interval')
n = 20
sample_mean = scores.sample(n, random_state=20).mean()[0]
st.markdown(
    r'สมมติว่าเราทำการสุ่มค่าจากข้อมูลผลการสอบเบื้องต้นที่มีค่าเฉลี่ย $\mu=53.15$ จากนศ.เพียง ${n}$ คนจะได้ค่า $\overline{{x}}={sample_mean}$'.format(
        sample_mean=sample_mean, n=n))
st.markdown(
    r'เนื่องจากเราทราบดีว่าค่าดังกล่าวมาจากการสุ่มตัวอย่างที่มีความผิดพลาดได้จาก $sampling\ error$ โดยค่าผิดพลาดมาตรฐาน $\sigma={std:.4f}$ ซึ่งสามารถคำนวณหาค่า $SE$ ได้ดังนี้'.format(
        std=scores.std()[0]))
st.markdown(r'$SE = \frac{\sigma}{\sqrt{n}} = \frac{10.66}{\sqrt{20}} = 2.383$')
normal_dist = Image.open('images/norm_dist.jpeg')

st.image(normal_dist)
st.markdown(
    r'จากภาพจะเห็นได้ว่า $95\%$ ของค่า $sample\ mean$ อยู่ในช่วงประมาณ $2\sigma$ ซึ่งเราคำนวณหาค่าที่แม่นยำได้จากสูตร')
st.code('NORM.S.INV(0.975) = 1.959 ~ 1.96')
st.code('NORM.S.INV(0.025) = -1.959 ~ -1.96')
st.markdown(
    r'สังเกตเราใช้ค่า $0.975$ แทน $0.95$ เนื่องจากเราต้องการค่าความมั่นใจ $95\%$ เมื่อแบ่งสองข้างของการกระจายตัวจะตกข้างละ $2.5\%$ ทางซ้ายและทางขวา')
st.markdown(r'ซึ่งค่า $Z-statistic$ จากความน่าจะเป็นเท่ากับ $97.5\%$ และ $2.5\%$ คือ $1.96$ และ $-1.96$ ตามลำดับ')
st.markdown(r'ค่า $95\%\ confidence\ interval$ คำนวณได้จาก')
upper_bound = 1.96 * 2.383 + sample_mean
lower_bound = -1.96 * 2.383 + sample_mean
st.markdown(
    r'$1.96SE + \overline{{x}} = 1.96\cdot2.383 + {sample_mean} = {upper_bound}$'.format(sample_mean=sample_mean,
                                                                                         upper_bound=upper_bound))
st.markdown(
    r'$-1.96SE + \overline{{x}} = -1.96\cdot2.383 + {sample_mean} = {lower_bound}$'.format(sample_mean=sample_mean,
                                                                                           lower_bound=lower_bound))
st.markdown(r'คำตอบคือ $95\%\ CI = ({:.2f} - {:.2f})$'.format(lower_bound, upper_bound))


def ci_norm_calculater(ci_alpha, ci_sigma, ci_sample_size, ci_mean):
    upper_percent = 1 - (ci_alpha / 2.0)
    z_stat = norm.ppf(upper_percent)
    se = ci_sigma / np.sqrt(ci_sample_size)
    return z_stat * se + ci_mean, (-1 * z_stat) * se + ci_mean, z_stat * se


ci_alpha_demo = st.number_input(r'$\alpha$', key='alpha_demo')


def plot_ci_from_scores(alpha, sigma, sample_size, mu):
    outs = []
    fig, ax = plt.subplots()
    for i in range(100):
        sample_mean = scores.sample(sample_size).mean()[0]
        upper, lower, _ = ci_norm_calculater(alpha, sigma, sample_size, sample_mean)
        color = 'blue' if lower < mu < upper else 'red'
        if color == 'red':
            outs.append(color)
        ax.hlines(y=random.randint(1, 30) + (random.randint(30, 40) / 10.0), xmin=lower, xmax=upper, color=color)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 40)

    ax.vlines(ymin=0, ymax=40, x=mu, color='green')

    st.pyplot(fig)
    st.markdown(r'Out of mean range = ${}$'.format(len(outs)))


plot_ci_from_scores(ci_alpha_demo, scores.std()[0], 20, mu=mean_scores)

st.header('Self Check')
st.markdown(r'1) ค่า $CI$ เท่าใดมีช่วงกว้างกว่ากันระหว่าง $95\%CI$ กับ $90\%CI$')
with st.expander('See an answer'):
    st.markdown(r'ค่า $95\%CI$ มีค่ากว้างกว่าค่า $90\%CI$')

st.header('Confidence Interval Calculator')
ci_alpha = st.number_input(r'$\alpha$')
ci_sigma = st.number_input(r'$SD$')
ci_mean = st.number_input(r'$\overline{x}$')
ci_sample_size = st.number_input('Sample size', key='ci_sample_size')
ci_upper_bound, ci_lower_bound, factor = ci_norm_calculater(ci_alpha, ci_sigma, ci_sample_size, ci_mean)
st.markdown(r'${}\%CI$'.format((1-ci_alpha) * 100))
st.markdown(r'$Lower bound = {:.2f}$'.format(ci_lower_bound))
st.markdown(r'$Upper bound = {:.2f}$'.format(ci_upper_bound))

st.markdown('$CI$ สามารถคำนวณได้จากสูตรใน Excel ดังนี้')
st.code('CONFIDENCE.NORM(alpha, std, sample_size) = CONFIDENCE.NORM({}, {}, {}) = {:.2f}'.format(ci_alpha, ci_sigma, ci_sample_size, factor))
st.markdown(r'ค่า $CI = {:.2f}\pm{:.2f}$'.format(ci_mean, factor))


st.header('Self Check')
st.markdown(r'1) น้ำหนักแรกเกิดของทารก $100$ คนคือ $3.0kg$ และมีค่า $SD=0.25kg$ ถ้าการกระจายตัวของน้ำหนักเป็นแบบ normal distribution จงหาค่า $95\%CI$ ของค่า $\overline{x}$')
with st.expander('See an answer'):
    st.markdown(r'$95\%CI = (2.95 - 3.05)$')


st.markdown(r'2) ตัวอย่างถุงลูกอมจำนวน $16$ ถุงถูกเลือกมาอย่างสุ่ม โดยการกระจายตัวของค่าน้ำหนักของถุงเป็นแบบปกติ ค่าน้ำหนักเฉลี่ยของถุงตัวอย่างคือ $2.0oz$ โดยมีค่า $s=0.12oz$ และค่า $\sigma=0.1oz$ จงหาค่า $95\%CI$ ของค่าเฉลี่ยน้ำหนักถุง')
with st.expander('See an answer'):
    st.markdown(r'$95\%CI = (1.95 - 2.05)$')

st.markdown(r'3) ครูท่านหนึ่งต้องการเก็บข้อมูลจำนวนจดหมายที่นร.ที่เข้าค่ายส่งกลับบ้าน โดยมีค่าเบี่ยงเบนมาตรฐานอยู่ที่ 2.5 ฉบับ จากการสำรวจนร.จำนวน 20 คนพบว่ามีค่าเฉลี่ยที่ 7.9 และมีค่า $s=2.8$ จงหาค่า $95\%CI$')
with st.expander('See an answer'):
    st.markdown(r'$95\%CI = (6.80 - 9.00)$')

st.markdown(r'''
## Hypothesis Testing

การทดสอบสมมติฐานทางสถิติมีแนวทางดังนี้

1. สร้างสมมติฐานที่สามารถพิสูจน์ได้
2. กำหนด _null hypothesis_
3. ตัดสินใจเลือกใช้การทดสอบทางสถิติ เก็บข้อมูลและคำนวณ
4. แปลและสรุปผล


สมมติว่าบริษัทยาแห่งหนึ่งต้องการทดสอบประสิทธิภาพของยาลดความดันโลหิตว่าดีกว่ายาที่ใช้ในปัจจุบันในผู้ป่วยที่มีคุณลักษณะคล้ายกัน เราสามารถตั้งสมมติฐานได้ว่าผู้ป่วยที่ได้รับการรักษาด้วยยาชนิดใหม่ (drug X) มีความดันโลหิตลดลงมากกว่าผู้ป่วยที่รักษาด้วยยาเดิม (drug Y) หากเรากำหนดให้ $\mu_{1}$ แทนค่าเฉลี่ยของความดันโลหิตที่ลดลงของผู้ป่วยที่ใช้ยา X และ $\mu_{2}$ แทนค่าความดันโลหิตที่ลดลงของผู้ป่วยที่ใช้ยา Y เราสามารถกำหนด null hypothesis ได้ดังนี้

$H_{0}: \mu_{1}\le\mu_{2}$

ดังนั้น alternative hypothesis ($H_{A}$ หรือ $H_{1}$) คือ

$H_{A}: \mu_{1}\gt\mu_{2}$


    null hypothesis และ alternative hypothesis ต้องเป็นแบบ mutually exclusive เท่านั้นคือไม่มีส่วนที่ซ้อนทับกัน เป็นได้อย่างได้อย่างหนึ่ง และผลลัพธ์ต้องเป็นแค่สองสมมติฐานนี้เท่านั้นไม่มีอย่างอื่น (exhaustive)

จากตัวอย่างนี้เราระบุว่าการจะปฏิเสธ (reject) null hypothesis นั้นค่า $\mu_{1}$ ต้องมากกว่าค่า $\mu_{2}$ เท่านั้น ซึ่งในกรณีนี้เราเรียก alternative hypothesis ว่าเป็นแบบ **_single-tailed_**

หากในอีกกรณีหนึ่งเราต้องการทดสอบแค่ว่าค่าการลดลงของความดันโลหิตในสองกลุ่มนั้นต่างกัน ไม่ว่ามากกว่าหรือน้อยกว่า เราจะตั้งสมมติฐานดังนี้


$H_{0}: \mu_{1}=\mu_{2}$

$H_{A}: \mu_{1}\neq\mu_{2}$

ซึ่งในกรณีนี้ alternative hypothesis จะเป็นแบบ **_two-tailed_** ซึ่งพบได้บ่อยในการทดสอบทางสถิติ

หลังจากการทดสอบจากการคำนวณจากข้อมูลที่รวบรวมได้แล้วเราสามารถตัดสินใจได้สองอย่างคือ

1. ปฏิเสธ null hypothesis
2. ไม่สามารถปฏิเสธ null hypothesis ได้

การที่เราไม่สามารถปฏิเสธ null hypothesis ได้นั้นไม่ใช่ว่า null hypothesis เป็นจริงหากแต่ว่าการศึกษาของเราไม่สามารถปฏิเสธได้เท่านั้นและโดยทั่วไปการที่เราปฏิเสธ null hypothesis นั้นก็ต่อเมื่อเราพบว่าความแตกต่างของค่าที่เราเปรียบเทียบนั้นมีนัยสำคัญทางสถิติหรือ **_statistically significant_** หรืออีกนัยหนึ่งคือผลที่ได้นั้นไม่ได้เกิดจากความบังเอิญ (randomness) ซึ่งก่อนการทดสอบทางสถิติเราควรจะกำหนดค่าระดับความน่าจะเป็นหรือ _p-value_ ที่ใช้เป็นตัวกำหนดการปฏิเสธ null hypothesis ซึ่งเรามักจะเห็นว่าเป็นค่า $p\lt0.05$ โดยทั่วไป ทั้งนี้เราอาจจะพบค่า p-value อื่นเช่น $p\lt0.01$ หรือ $p\lt0.001$ ได้เช่นกัน

การทดสอบทางสถิตินั้นเป็นหลักการที่อาศัยความน่าจะเป็นดังนั้นย่อมมีโอกาสที่จะผิดพลาด ซึ่งนักสถิติได้แบ่งออกเป็นสองแบบคือ

||||True state of population |
| --- | --- | --- | --- |
||| $H_{0}$ true | $H_{A}$ true |
|Decision based on sample statistics|Fail to reject $H_{0}$| Correct desicion: $H_{0}$ true and $H_{0}$ not rejected | Type II error or $\beta$
||Reject $H_{0}$|Type I error or $\alpha$|Correct decision:$H_{0}$ false and $H_{0}$ rejected|

การที่เรากำหนดค่า $\alpha=0.05$ หมายถึงเรายอมรับว่ามีความน่าจะเป็นร้อยละ $5\%$ ที่จะเกิดความผิดพลาดเพราะเราปฏิเสธ $H_{0}$ ในขณะที่เราไม่สามารถจะปฏิเสธ $H_{0}$ ได้ $(failed\ to\ reject\ H_{0}$) 

#### ตัวอย่าง

สมมติเราต้องการทดสอบสมมติฐานว่าเหรียญที่มีสองด้านเป็นเหรียญที่มีมาตรฐานหรือไม่ หมายความว่าโอกาสที่จะโยนเหรียญแต่ละครั้งแล้วได้หัวหรือก้อยคือ $0.5$ หากว่าเราทดลองโยนเหรียญ $10$ ครั้งแล้วพบว่าได้หัว $8$ ครั้งเหรียญนี้จะได้มาตรฐานหรือไม่

#### วิธีคิด

ปัญหาข้อนี้เราสามารถเปรียบเทียบความน่าจะเป็นของการได้หัว $\frac{8}{10}$ ครั้งได้โดยการเปรียบเทียบจากความน่าจะเป็นของการกระจายตัวแบบ binomial distribution ซึ่งเมื่อเราคำนวณแล้วพบว่าโอกาสที่จะได้หัว $\frac{8}{10}$ ครั้งคือ $0.0439$ ทั้งนี้เนื่องจากเราต้องการทดสอบว่าเหรียญได้มาตรฐานหรือไม่ดังนั้นเราสามารถคำนวณหาโอกาสที่จะได้หัวอย่างน้อย $8$ ครั้งคือ $8,9$ และ $10$ ได้เท่ากับ $0.0439 + 0.0098 + 0.0010 = 0.0547$ โดย $P(h)=0.5$ ดังนั้นในกรณีนี้เราไม่สามารถสรุปได้ว่าเหรียญไม่ได้มาตรฐานเพราะความน่าจะเป็นหรือ $p-value\gt0.05$
''')