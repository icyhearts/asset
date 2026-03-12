
#include <atomic>
#include <cstdio>
#include <utility>

template <typename T>
class SharedPtr {
  public:
    SharedPtr(T *ptr = nullptr) {
    }
  private:
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
