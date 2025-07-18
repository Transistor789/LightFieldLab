#include <cstdio>
#include <functional>
#include <future>
#include <iostream>
#include <ostream>
#include <thread>
#include <utility>
#include <vector>

int future_get_helper() {
	std::this_thread::sleep_for(std::chrono::seconds(1));
	std::cout << "1秒已过" << std::endl;
	return 999;
}
void future_get() {
	// 启动异步任务，用async包装函数
	std::future<int> answerFuture =
		std::async(std::launch::async, future_get_helper);

	// 这里可以继续做其他事情...
	std::cout << "正在计算答案，请稍等..." << std::endl;
	for (int i = 0; i < 10; i++) {
		std::cout << i << " " << std::flush;
	}
	// 获取结果（如果还没计算完，会等待）
	int answer = answerFuture.get();
	for (int i = 50; i < 50 + 10; i++) {
		std::cout << i << " " << std::flush;
	}
	std::cout << "\n答案是: " << answer << std::endl;
}
void future_wait_helper(std::vector<int> &arr) {
	for (int i = 0; i < arr.size(); i++) {
		arr[i] += 100; // 每个元素加100
	}
	std::this_thread::sleep_for(std::chrono::seconds(2));
	std::cout << "2秒已过" << std::endl;
}
void future_wait() {
	std::vector<int> arr(10);
	for (int i = 0; i < 10; i++) {
		arr[i] = i;
		printf("%d ", arr[i]);
	}
	std::future<void> answerFuture =
		std::async(std::launch::async, future_wait_helper, std::ref(arr));

	answerFuture.wait();
	for (int i = 0; i < 10; i++) {
		printf("%d ", arr[i]);
	}
	std::cout << "计算结束" << std::endl;
}
int future_promise_helper(int x, int y) {
	std::this_thread::sleep_for(std::chrono::seconds(3));
	std::cout << "3秒已过" << std::endl;
	return x + y;
}
void future_promise() {
	std::promise<int> promise;
	std::future<int> future = promise.get_future();

	std::thread worker([&promise]() {
		int result = future_promise_helper(1, 2);
		promise.set_value(result);
	});
	std::cout << "主线程等待结果..." << std::endl;
	int result = future.get();
	std::cout << "答案是: " << result << std::endl;
	worker.join();
}
void compute_pi(const long num_steps, std::promise<double> &&promise) {
	std::cout << "compute_pi thread :" << std::this_thread::get_id()
			  << std::endl;
	double step = 1.0 / num_steps;
	double sum = 0.0;
	for (long i = 0; i < num_steps; i++) {
		double x = (i + 0.5) * step;
		sum += 4.0 / (1.0 + x * x);
	}
	promise.set_value(sum * step);
}
void display(std::future<double> &&future) {
	std::cout << "display thread :" << std::this_thread::get_id() << std::endl;
	double pi = future.get();
	std::cout << "返回值：" << pi << std::endl;
}
void print() {
	std::cout << "print thread :" << std::this_thread::get_id() << std::endl;
}
void future_promise_2thread() {
	std::cout << "future_promise_2thread thread :" << std::this_thread::get_id()
			  << std::endl;
	print();
	const long int num_steps = 100000000;
	std::promise<double> promise;
	std::future<double> future = promise.get_future();
	std::thread t1(compute_pi, num_steps, std::move(promise));
	std::thread t2(display, std::move(future));
	t1.detach();
	t2.join();
	// t1.join();
	// t2.join();
}
void test() {
	std::vector<int> arr(10);
	for (int i = 0; i < 10; i++) arr[i] = i;

	// 1. 启动promise任务（3秒）
	std::promise<int> promise;
	std::future<int> f1 = promise.get_future();
	std::thread worker([&promise]() {
		int result = future_promise_helper(1, 2);
		promise.set_value(result); // 必须设置值！
	});

	// 2. 启动async任务（2秒）
	std::future<void> f2 =
		std::async(std::launch::async, future_wait_helper, std::ref(arr));

	// 3. 启动async任务（1秒）
	std::future<int> f3 = std::async(std::launch::async, future_get_helper);

	// 明确等待所有任务完成（按耗时从短到长）
	std::cout << "最短任务结果: " << f3.get() << std::endl; // 1秒
	f2.wait();												// 2秒
	std::cout << "最长任务结果: " << f1.get() << std::endl; // 3秒

	// 打印修改后的数组
	std::cout << "修改后的数组: ";
	for (int num : arr) std::cout << num << " ";
	std::cout << std::endl;

	worker.join(); // 虽然f1.get()已保证完成，但最好显式join
}

int main() {
	// future_get();
	// future_wait();
	// future_promise();
	// test();
	future_promise_2thread();
	return 0;
}