#include <future>
#include <iostream>
#include <memory>
#include <tuple>
#include <type_traits>
class Test {
public:
	void inClassMethod() { std::cout << "inClassMethod called!\n"; }
	static void staticMethod() { std::cout << "staticMethod called!\n"; }
	int add(int a, int b) { return a + b; }
};
void outOfClassMethod() { std::cout << "outOfClassMethod called!\n"; }
static void staticOutOfClassMethod() {
	std::cout << "staticOutOfClassMethod called!\n";
}
// char   add(char a, char b);
// int	   add(int a, int b);
// float  add(float a, float b);
// double add(double a, double b);
template <typename T, typename U>
auto add(T t, U u) -> decltype(t + u) { // 返回类型由 t + u 的结果决定
	return t + u;
}
void test_func_ptr() {
	std::cout << "test_func_ptr called!\n";
	void (*pfunc1)() = &outOfClassMethod;
	void (*pfunc2)() = &Test::staticMethod;

	Test test;
	Test *pclass = new Test();
	std::unique_ptr<Test> puclass = std::make_unique<Test>();
	void (Test::*pfunc3)() = &Test::inClassMethod;
	void (*pfunc4)() = &Test::staticMethod;
	void (*pfunc5)() = &staticOutOfClassMethod;

	pfunc1();
	pfunc2();

	(test.*pfunc3)();
	(pclass->*pfunc3)();
	(puclass.get()->*pfunc3)();

	pfunc4();
	pfunc5();

	delete pclass;
}
void test_decltype() {
	std::cout << "test_decltype called!\n";
	int num = 10;
	decltype(num) num2 = 20;
	// auto n; // auto 必须初始化 decltype不用
	auto sum = num + num2;
	std::cout << sum << std::endl;

	// using resultype1 = ;
	std::cout << std::is_void_v<decltype(outOfClassMethod())> << std::endl;
	std::cout << std::is_void_v<decltype(add(0, 0))> << std::endl;
	std::cout << typeid(decltype(add(' ', ' '))).name() << std::endl;
	std::cout << typeid(decltype(add(0, 0))).name() << std::endl;
	std::cout << typeid(decltype(add(1.0f, 2.0f))).name() << std::endl;
	std::cout << typeid(decltype(add(3.0, 4.0))).name() << std::endl;

	// std::cout << typeid().name() << std::endl;
}
void printPersonInfo(const std::string &name, int age, double height) {
	std::cout << "Name: " << name << ", Age: " << age << ", Height: " << height
			  << "m\n";
}
void test_tuple_apply() {
	std::cout << "test_tuple_apply called!\n";

	auto person = std::make_tuple("Alice", 30, 1.75);
	std::apply(printPersonInfo, person);

	auto args = std::make_tuple(1, 2);

	int (*addPtr)(int, int) = &add<int, int>;
	int (Test::*addPtrInClass)(int, int) = &Test::add;
	void (Test::*inClassMethodPtrInClass)() = &Test::inClassMethod;
	void (*staticMethodInClass)() = &Test::staticMethod;

	Test test;
	Test *testPtr = new Test();
	std::unique_ptr<Test> testUniquePtr = std::make_unique<Test>();

	std::cout << std::apply(addPtr, args) << std::endl;
	std::cout << std::apply(
		[&](auto... params) { return (test.*addPtrInClass)(params...); }, args)
			  << std::endl;
	std::cout << std::apply(
		[&](auto... params) { return (testPtr->*addPtrInClass)(params...); },
		args) << std::endl;
	std::cout << std::apply(
		[&](auto... params) {
			return (testUniquePtr.get()->*addPtrInClass)(params...);
		},
		args) << std::endl;

	std::apply([&]() { (test.*inClassMethodPtrInClass)(); }, std::make_tuple());
	std::apply([&]() { (testPtr->*inClassMethodPtrInClass)(); },
			   std::make_tuple());
	std::apply([&]() { (testUniquePtr.get()->*inClassMethodPtrInClass)(); },
			   std::make_tuple());

	std::apply([]() { Test::staticMethod(); }, std::make_tuple());
	std::apply([&]() { staticMethodInClass(); }, std::make_tuple());
}
void test_async() {
	class Worker {
	public:
		using Callback = std::function<void(int)>;
		std::future<void> run(int num, Callback callback) {
			return std::async(std::launch::async, [&]() {
				for (int i = 0; i < 1000; i++) {
					num++;
				}
				callback(num); // 在子线程中调用回调
			});
		}
	} worker;

	auto future = worker.run(
		24, [](int num) { std::cout << "Result: " << num << std::endl; });

	for (int i = 0; i < 10; i++) {
		std::cout << i << " ";
	}

	future.wait(); // 等待异步任务完成（可选）
}
int main(int argc, char *argv[]) {
	// std::cout << "Hello, world!\n";
	// test_func_ptr();
	// test_decltype();
	// test_tuple_apply();
	test_async();
	return 0;
}