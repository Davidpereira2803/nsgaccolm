import random

def estimate_average_completion_time(matrix):
    # Estimate average completion time per job by summing all processing times and dividing by the number of jobs
    average = 0
    n_machines, n_jobs = matrix.shape
    for i in range (n_jobs):
        for j in range (n_machines):
            average += matrix[j][i]
    average /= n_jobs
    return float(average)

def compute_due_dates(matrix, P, R, T):
    # Generate due dates using parameters P (average completion time), R (randomness range), and T (tightness factor). Each due date is computed as P * (1 - T - R/2 + random_uniform(0, R)).
    n_machines, n_jobs = matrix.shapes
    due_dates = []
    for i in range (n_jobs):
        due_date = P * (1-T-(R/2) + random.uniform(0,R))
        due_dates.append(due_date)
    return due_dates