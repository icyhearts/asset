
#include <atomic>
#include <cstdio>
#include <utility>

template <typename T>
class SharedPtr {
  public:
    SharedPtr(T *ptr = nullptr): ptr_(ptr), ref_count_(ptr_ ? new std::atomic<int>(1): nullptr) {
    }
    // copy ctor
    SharedPtr(const SharedPtr &other) {
      ptr_ = other.ptr_;
      ref_count_ = other.ref_count_;
      if (ref_count_) {
        ref_count_->fetch_add(1);
      }
    }
    // move ctor
    SharedPtr(SharedPtr &&other) {
      ptr = other.ptr_;
      ref_count_ = other.ref_count_;
      other.ptr_ = nullptr;
      other.ref_count_ = nullptr;

    }
    // copy assign
    SharedPtr &SharedPtr(const SharedPtr &other) {
      if (this != &other) {
        release();
        ptr_ = other.ptr_;
        ref_count_ = other.ref_count_;
        if (ref_count_) {
          ref_count_->fetch_add(1);
        }
      }
      return *this;

    }
    // move asign
    SharedPtr &SharedPtr(SharedPtr &&other) {
      if (this != &other) {
        release();
        ptr_ = other.ptr_;
        ref_count_ = other.ref_count_;

        other.ptr_ = nullptr;
        other.ref_count_ = nullptr;
      }
      return *this;
    }
    ~SharedPtr () {
      release();
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
