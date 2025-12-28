#ifndef STREAMREDIRECTOR_H
#define STREAMREDIRECTOR_H

#include <functional>
#include <iostream>
#include <streambuf>
#include <string>

// 定义回调类型

class StreamRedirector : public std::streambuf {
public:
	using LogCallback = std::function<void(const std::string &)>;

	// 构造函数：传入原本的流（如 cout）和回调函数
	StreamRedirector(std::ostream &stream, LogCallback cb)
		: m_stream(stream), m_callback(cb) {
		// 1. 保存原本的缓冲区 (以便析构时恢复)
		m_oldBuf = stream.rdbuf();
		// 2. 将流的缓冲区替换为 this (当前类)
		stream.rdbuf(this);
	}

	~StreamRedirector() {
		// 3. 恢复原本的缓冲区
		if (!m_buffer.empty()) {
			flush(); // 析构前把剩下的发出去
		}
		m_stream.rdbuf(m_oldBuf);
	}

protected:
	// 核心重载：每当流接收到一个字符时调用此函数
	int_type overflow(int_type c) override {
		if (c != EOF) {
			char ch = static_cast<char>(c);
			if (ch == '\n') {
				flush(); // 遇到换行，发送日志
			} else {
				m_buffer += ch; // 没换行，先攒着
			}
		}
		return c;
	}

	// 也可以重载 xsputn 提高批量输出的性能，但 overflow 够用了

private:
	void flush() {
		if (!m_buffer.empty() && m_callback) {
			m_callback(m_buffer);
			m_buffer.clear();
		}
	}

	std::ostream &m_stream;	  // 被劫持的流 (cout/cerr)
	std::streambuf *m_oldBuf; // 原本的流缓冲区
	LogCallback m_callback;	  // 日志回调
	std::string m_buffer;	  // 内部暂存区
};

#endif // STREAMREDIRECTOR_H