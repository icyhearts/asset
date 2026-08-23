#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;

template <typename T>
void PrintVector(vector<T> &vec, const string &title = "vector:") {

  cout << title;
  int N = vec.size();
  for (int i = 0; i < N; ++i) {
    cout << vec[i] << ",";
  }
  cout << endl;
}

void get_next_tile_vert(const int *cu_tiles_ptr, int iblock,
                                                   int num_group, int &igroup, int &itile_m,
                                                   int &itile_n, int total_m) {
  int itile_m_total = iblock % total_m;
  itile_n = iblock / total_m;

  int left = 0;
  int right = num_group;
  while (left <= right) {
    int mid = left + (right - left) / 2;
    if (cu_tiles_ptr[mid] > itile_m_total) {
      right = mid - 1;
    } else {
      left = mid + 1;
    }
  }
  itile_m = itile_m_total - cu_tiles_ptr[right];
  igroup = right;
  cout << "iblock:" << iblock << ",itile_m_total:" << itile_m_total << ", result:" << ", cu_tiles_ptr[right]:" << cu_tiles_ptr[right]
    << ", igroup: " << right << ",itile_m:" << itile_m << ",itile_n:" << itile_n << "\n";

}

void get_next_tile_vert(const vector<int> &cu_tiles_ptr, int iblock,
                                                   int num_group, int &igroup, int &itile_m,
                                                   int &itile_n, int total_m) {
  int itile_m_total = iblock % total_m;
  itile_n = iblock / total_m;

  auto find_it = std::upper_bound(cu_tiles_ptr.begin(), cu_tiles_ptr.end(), itile_m_total);
  int find_idx = std::distance( cu_tiles_ptr.begin(), find_it);
  int right = find_idx - 1;
  itile_m = itile_m_total - cu_tiles_ptr[right];
  igroup = right;
  cout << "iblock:" << iblock << ",itile_m_total:" << itile_m_total << ", result:" << ", cu_tiles_ptr[right]:" << cu_tiles_ptr[right]
    << ", igroup: " << right << ",itile_m:" << itile_m << ",itile_n:" << itile_n << "\n";

}

int main(int argc, char **argv) {
  int igroup, itile_m, itile_n;
  int igroup2, itile_m2, itile_n2;
  int total_m = 1000;

#if 0
  {
  vector<int> cu_tiles_ptr = {1, 3, 5, 7, 9, 11};
  PrintVector(cu_tiles_ptr);
  int len = cu_tiles_ptr.size();
  int num_group = len - 1;
  for (int iblock = 0; iblock <= num_group; iblock++) {
    get_next_tile_vert(cu_tiles_ptr.data(), iblock,
        num_group, igroup, itile_m,
        itile_n, total_m);
  }

  }
#endif

  cout << "----------------------------------------\n";

  {
  vector<int> cu_tiles_ptr = {0, 1, 2, 3, 5, 7, 9, 12, 15};
  PrintVector(cu_tiles_ptr);
  total_m = 15;
  int len = cu_tiles_ptr.size();
  int num_group = len - 1;
  for (int iblock = 0; iblock <= 30; iblock++) {
    get_next_tile_vert(cu_tiles_ptr.data(), iblock,
        num_group, igroup, itile_m,
        itile_n, total_m);

    get_next_tile_vert(cu_tiles_ptr, iblock,
        num_group, igroup2, itile_m2,
        itile_n2, total_m);
    bool all_eq = (igroup == igroup2) && (itile_m == itile_m2) && (itile_n == itile_n2);
    cout << "all_eq:" << all_eq << "\n";
  }
  }
}
