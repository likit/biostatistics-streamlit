import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from PIL import Image

scores = pd.read_csv('datasets/exam1.csv', header=None)

st.title('Welcome to Biostatistics Learning Platform!')
st.text('Developed by Asst. Prof.Likit Preeyanon, Ph.D.')

st.header('Central Limit Theorem!')
st.text('Let say we collect scores of a Parasitology exam from 80 students and list them below.')
st.write(scores)

mean_scores = scores.mean()[0]

st.text('Descriptive Statistics')
st.write(scores.describe())

fig, ax = plt.subplots()
ax.hist(scores, bins=20)

st.pyplot(fig)

st.markdown(
    'What will happen if we select scores from a number of random students from this same class to calculate a mean for multiple times?')


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

    axes[0].set_title('Deviation From The True Mean')

    st.pyplot(fig)

    st.markdown('Standard deviation of the sample means is {}'.format(np.std(sample_means)))
    st.markdown('Standard Error of the Mean (SE) is {}'.format(np.std(scores)[0] / np.sqrt(num_sample)))


num_sample = st.slider('Number of students', min_value=10, max_value=70, step=10)
num_trials = st.slider('Number of trials', min_value=50, max_value=1000, step=50)

plot_deviation_sample_mean_chart(scores_mean=mean_scores, num_sample=num_sample, num_trials=num_trials)


def plot_se(sigma, n=[10, 20, 40, 80, 160, 320, 640, 1280]):
    ses = sigma / np.sqrt(n)
    st.dataframe({'SE': ses, 'n': n})


sigma = st.slider('Sigma', min_value=50, max_value=100, step=10)

plot_se(sigma)

st.header('Z statistics (not Z-score)')
st.markdown('***$Z statistics$*** คำนวณจากสมการ')
st.markdown(r'$Z = \frac{\overline{x}-\mu}{\frac{\sigma}{\sqrt{n}}}$')
st.markdown(
    r'ค่า ***$Z score$*** ($\frac{\overline{x}-\mu}{\sigma}$) มีค่าเท่ากับ $Z statistics$ เมื่อ $n=1$.')
st.markdown(
    r'สมมติเราสุ่มตัวอย่างจากประชากรที่มีค่า $\mu=50$ และ $\sigma=10$ มาสามกลุ่ม โดยทั้งสามกลุ่มมีค่า ${\overline{x}}=52$ และจำนวนตัวอย่างคือ $30, 60, 100$ ตามลำดับ จงคำนวณค่า $Z statistics$')


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

st.markdown('Z statistics = {}'.format(calculate_z_stat(pop_mean, sigma, n, sample_mean)))
st.markdown('SEM = {}'.format(sigma / np.sqrt(n)))
#plot_norm_pdf(pop_mean, sigma, n)
plot_z_score(sample_mean, pop_mean, sigma, n)

st.markdown('เราสามารถคำนวณหาค่าความน่าจะเป็นของค่า Z statistics ในโปรแกรม Excel ได้จากสูตร')
st.code(f'NORM.S.DIST({zscore:.2f}, TRUE) = {norm.cdf(zscore):.2f}')
st.markdown(f'หมายความว่าโอกาสที่จะได้ค่าต่ำกว่าหรือเท่ากับ {sample_mean} ({zscore:.2f}SD) คือ {norm.cdf(zscore)*100:.2f}%')
st.markdown(f'หมายความว่าโอกาสที่จะได้ค่ามากกว่า {sample_mean} ({zscore:.2f}SD) คือ {(1 - norm.cdf(zscore))*100:.2f}%')

st.markdown('ในทางกลับกันเราสามารถคำนวณหา Z statistics จากความน่าจะเป็นได้ในโปรแกรม Excel ได้จากสูตร')
st.code(f'NORM.S.INV({norm.cdf(zscore):.2f}) = {norm.ppf(norm.cdf(zscore)):.2f}')

normal_dist = Image.open('images/norm_dist.jpeg')

st.image(normal_dist, caption='Normal Distribution')