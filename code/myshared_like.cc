
#include <atomic>
#include <cstdio>
#include <utility>

template <typename T>
class SharedPtr {
  public:
    SharedPtr(T *ptr = nullptr): ptr_(ptr), ref_count_(ptr_ ? new std::atomic<int>(1): nullptr) {
      ptr_
    }
    // copy ctor
    SharedPtr(const SharedPtr &other) {

    }
    // move ctor
    SharedPtr(const SharedPtr &&other) {
    }
    // copy assign
    SharedPtr &SharedPtr(const SharedPtr &other) {

    }
    // move asign
    SharedPtr &SharedPtr(const SharedPtr &&other) {
    }
  private:
    void release() {
      if (ref_count_) {
        delete ref_count_;
        delete ptr_;
      }
      ref_count_ = nullptr;
      ptr_ = nullptr;
    }
    T* ptr_;
    std::atomic<int>* ref_count_;
};
struct Foo {
    int val;
    Foo(int v) : val(v) { printf("  Foo(%d) created\n", val); }
    ~Foo() { printf("  Foo(%d) destroyed\n", val); }
};

int main () {
}
