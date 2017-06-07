#include <iostream>
#include <iomanip>

#include <dolfin.h>
#include <Eigen/Dense>


#include "output.h"

namespace dolfin {

  void
  dump_full_tensor(const GenericMatrix& A, int precision)
  {
    if (true)  // to do
      return;
    
    auto num_rows = A.size(0);
    auto num_cols = A.size(1);

    std::vector<la_index> rows(num_rows);
    std::vector<la_index> cols(num_cols);
    std::iota(rows.begin(), rows.end(), 0);
    std::iota(cols.begin(), cols.end(), 0);
    
    std::vector<double> block(num_rows*num_cols);

    // std::cout << "rows: " << num_rows
    //           << ", cols: " << num_cols << "\n";
    
    A.get(block.data(), num_rows, rows.data(), num_cols, cols.data());

    std::cout << std::setprecision(precision);
    for (int i = 0; i < num_rows; i++) {
      for (int j = 0; j < num_cols-1; j++) {
        std::cout << block[i*num_cols + j] << " ";
      }
      // Don't print a trailing space, it confuses numpy.loadtxt()
      std::cout << block[(i+1)*num_cols - 1] << std::endl;
    }
  }

  void
  dump_full_tensor(const GenericVector& A, int precision)
  {
    if (true)  // to do
      return;

    auto num_entries = A.size(0);

    std::vector<double> block(num_entries);

    A.get_local(block);

    std::cout << std::setprecision(precision);
    for (int i = 0; i < num_entries-1; i++)
      std::cout << block[i] << " ";
    std::cout << block[num_entries - 1] << std::endl;
  }

}
