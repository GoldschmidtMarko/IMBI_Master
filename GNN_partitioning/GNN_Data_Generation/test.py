import tqdm
import multiprocessing

def my_function(arg1, arg2, arg3):
  return arg1 + arg2 + arg3

def my_function_star(args):
    return my_function(*args)

if __name__ == '__main__':
    jobs = 8
    with multiprocessing.Pool(jobs) as pool:
        args = [(i, i, i) for i in range(100000)]
        list(tqdm.tqdm(pool.imap(my_function_star, args), total=len(args)))