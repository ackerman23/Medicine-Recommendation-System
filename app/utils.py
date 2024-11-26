import matplotlib.pyplot as plt

def plot_reviews(data):
    labels = ['Excellent', 'Average', 'Poor']
    sizes = [data['Excellent Review %'].mean(), data['Average Review %'].mean(), data['Poor Review %'].mean()]

    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.savefig('app/static/reviews.png')
