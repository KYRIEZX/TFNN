import numpy as np

def compute_loss(w,b,points):
    total_loss=0
    for i in range(len(points)):
        x=points[i,0]
        y=points[i,1]
        total_loss+=(w*x+b-y)**2
    return total_loss/float(len(points))

def step_gradient(w_current,b_current,points,learning_rate):
    w_gradient=0
    b_gradient=0
    N = float(len(points))
    for i in range(len(points)):
        x=points[i,0]
        y=points[i,1]
        w_gradient+=(2/N)*x*(w_current*x+b_current-y)
        b_gradient+=(2/N)*((w_current*x+b_current)-y)
    w_new=w_current - (w_gradient * learning_rate)
    b_new=b_current - (b_gradient * learning_rate)
    return [w_new,b_new]

def gradient_descent(w_init,b_init,learning_rate,points,num_interations):
    w=w_init
    b=b_init
    for i in range(num_interations):
        [w,b]=step_gradient(w,b,points,learning_rate)
    return [w,b]

def run():
    points = np.genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0  # initial y-intercept guess
    initial_w = 0  # initial slope guess
    num_iterations = 5000
    print("Starting gradient descent at b = {0}, w = {1}, error = {2}"
          .format(initial_b, initial_w,
                  compute_loss(initial_w, initial_b, points))
          )
    print("Running...")
    [w, b] = gradient_descent( initial_w, initial_b, learning_rate, points, num_iterations)
    print("After {0} iterations b = {1}, w = {2}, error = {3}".
          format(num_iterations, b, w,
                 compute_loss(w, b, points))
          )


if __name__ == '__main__':
    run()