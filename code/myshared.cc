#include <atomic>
#include <cstdio>
#include <utility>

template <typename T>
class SharedPtr {
public:
    // Constructor with raw pointer
    explicit SharedPtr(T* ptr = nullptr)
        : ptr_(ptr), ref_count_(ptr ? new std::atomic<int>(1) : nullptr) {
        printf("[Constructor] ptr=%p, ref_count=%d\n",
               (void*)ptr_, ref_count_ ? ref_count_->load() : 0);
    }

    // Copy constructor
    SharedPtr(const SharedPtr& other)
        : ptr_(other.ptr_), ref_count_(other.ref_count_) {
        if (ref_count_) {
            ref_count_->fetch_add(1);
        }
        printf("[CopyCtor] ptr=%p, ref_count=%d\n",
               (void*)ptr_, ref_count_ ? ref_count_->load() : 0);
    }

    // Move constructor
    SharedPtr(SharedPtr&& other) noexcept
        : ptr_(other.ptr_), ref_count_(other.ref_count_) {
        other.ptr_ = nullptr;
        other.ref_count_ = nullptr;
        printf("[MoveCtor] ptr=%p, ref_count=%d\n",
               (void*)ptr_, ref_count_ ? ref_count_->load() : 0);
    }

    // Copy assignment
    SharedPtr& operator=(const SharedPtr& other) {
        printf("[CopyAssign] begin\n");
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            ref_count_ = other.ref_count_;
            if (ref_count_) {
                ref_count_->fetch_add(1);
            }
        }
        printf("[CopyAssign] ptr=%p, ref_count=%d\n",
               (void*)ptr_, ref_count_ ? ref_count_->load() : 0);
        return *this;
    }

    // Move assignment
    SharedPtr& operator=(SharedPtr&& other) noexcept {
        printf("[MoveAssign] begin\n");
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            ref_count_ = other.ref_count_;
            other.ptr_ = nullptr;
            other.ref_count_ = nullptr;
        }
        printf("[MoveAssign] ptr=%p, ref_count=%d\n",
               (void*)ptr_, ref_count_ ? ref_count_->load() : 0);
        return *this;
    }

    // Destructor
    ~SharedPtr() {
        printf("[Destructor] ptr=%p, ref_count=%d\n",
               (void*)ptr_, ref_count_ ? ref_count_->load() : 0);
        release();
    }

    // Reset: release current, optionally take new pointer
    void reset(T* ptr = nullptr) {
        printf("[Reset] old_ptr=%p, new_ptr=%p\n", (void*)ptr_, (void*)ptr);
        release();
        ptr_ = ptr;
        ref_count_ = ptr ? new std::atomic<int>(1) : nullptr;
    }

    // Swap
    void swap(SharedPtr& other) noexcept {
        printf("[Swap] this->ptr=%p, other->ptr=%p\n",
               (void*)ptr_, (void*)other.ptr_);
        std::swap(ptr_, other.ptr_);
        std::swap(ref_count_, other.ref_count_);
    }

    T* get() const { return ptr_; }
    int use_count() const { return ref_count_ ? ref_count_->load() : 0; }

    T& operator*() const { return *ptr_; }
    T* operator->() const { return ptr_; }
    explicit operator bool() const { return ptr_ != nullptr; }

private:
    void release() {
        if (ref_count_ && ref_count_->fetch_sub(1) == 1) {
            delete ptr_;
            delete ref_count_;
        }
        ptr_ = nullptr;
        ref_count_ = nullptr;
    }

    T* ptr_;
    std::atomic<int>* ref_count_;
};

struct Foo {
    int val;
    Foo(int v) : val(v) { printf("  Foo(%d) created\n", val); }
    ~Foo() { printf("  Foo(%d) destroyed\n", val); }
};

int main() {
    printf("=== 1. Constructor ===\n");
    SharedPtr<Foo> p1(new Foo(1));
    printf("  p1.use_count=%d, val=%d\n\n", p1.use_count(), p1->val);

    printf("=== 2. Copy Constructor ===\n");
    SharedPtr<Foo> p2(p1);
    printf("  p1.use_count=%d, p2.use_count=%d\n\n", p1.use_count(), p2.use_count());

    printf("=== 3. Move Constructor ===\n");
    SharedPtr<Foo> p3(std::move(p2));
    printf("  p2.use_count=%d, p3.use_count=%d\n\n", p2.use_count(), p3.use_count());

    printf("=== 4. Copy Assignment ===\n");
    SharedPtr<Foo> p4(new Foo(4));
    p4 = p1;
    printf("  p1.use_count=%d, p4.use_count=%d, val=%d\n\n",
           p1.use_count(), p4.use_count(), p4->val);

    printf("=== 5. Move Assignment ===\n");
    SharedPtr<Foo> p5(new Foo(5));
    p5 = std::move(p3);
    printf("  p3.use_count=%d, p5.use_count=%d, val=%d\n\n",
           p3.use_count(), p5.use_count(), p5->val);

    printf("=== 6. Reset (with new pointer) ===\n");
    p5.reset(new Foo(6));
    printf("  p5.use_count=%d, val=%d\n\n", p5.use_count(), p5->val);

    printf("=== 7. Reset (to nullptr) ===\n");
    p5.reset();
    printf("  p5.use_count=%d\n\n", p5.use_count());

    printf("=== 8. Swap ===\n");
    SharedPtr<Foo> pa(new Foo(10));
    SharedPtr<Foo> pb(new Foo(20));
    printf("  before: pa.val=%d, pb.val=%d\n", pa->val, pb->val);
    pa.swap(pb);
    printf("  after:  pa.val=%d, pb.val=%d\n\n", pa->val, pb->val);

    printf("=== 9. Destructor (leaving scope) ===\n");
    return 0;
}
