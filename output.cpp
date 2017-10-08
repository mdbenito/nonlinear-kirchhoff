#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <dolfin.h>

#include "output.h"

namespace NLK {

  using namespace dolfin;
  int DEBUG = 0;
  
  void
  dump_full_tensor(const GenericMatrix& A, int precision,
                   const std::string& name, bool asfile, bool force)
  {
    if (!force && DEBUG < 2)
      return;
    
    auto num_rows = A.size(0);
    auto num_cols = A.size(1);

    std::vector<la_index> rows(num_rows);
    std::vector<la_index> cols(num_cols);
    std::iota(rows.begin(), rows.end(), 0);
    std::iota(cols.begin(), cols.end(), 0);
    
    std::vector<double> block(num_rows*num_cols);

    std::ostream* out = &std::cout;
    std::ofstream fs;
    if (name.size() > 0 && asfile)
    {
      fs.open(name, std::fstream::out | std::fstream::trunc);
      out = &fs;
    } else if (name.size() > 0 && !asfile)
      *out << name << ": ";
    
    A.get(block.data(), num_rows, rows.data(), num_cols, cols.data());

    *out << std::setprecision(precision);
    for (int i = 0; i < num_rows; i++) {
      for (int j = 0; j < num_cols-1; j++)
        *out << block[i*num_cols + j] << " ";
      // Don't print a trailing space, it confuses numpy.loadtxt()
      *out << block[(i+1)*num_cols - 1] << std::endl;
    }
  }

  void dump_full_tensor(std::shared_ptr<const dolfin::GenericMatrix> A,
                        int precision, const std::string& name,
                        bool asfile, bool force)
  {
    return dump_full_tensor(*A, precision, name, asfile);
  }
  
  void
  dump_full_tensor(const GenericVector& A, int precision,
                   const std::string& name, bool asfile, bool force)
  {
    if (!force && DEBUG < 2)
      return;
    
    auto num_entries = A.size(0);

    std::vector<double> block(num_entries);

    A.get_local(block);

    std::ostream* out = &std::cout;
    std::ofstream fs;
    if (name.size() > 0 && asfile)
    {
      fs.open(name, std::fstream::out | std::fstream::trunc);
      out = &fs;
    } else if (name.size() > 0 && !asfile)
      *out << name << ": ";
    
    *out << std::setprecision(precision);
    for (int i = 0; i < num_entries-1; i++)
      *out << block[i] << " ";
    *out << block[num_entries - 1] << std::endl;
  }

  void dump_full_tensor(std::shared_ptr<const dolfin::GenericVector> A,
                        int precision, const std::string& name,
                        bool asfile, bool force)
  {
    return dump_full_tensor(*A, precision, name, asfile, force);
  }

  void dump_raw_matrix(const double* A, int m, int n, int precision,
                       const std::string& name, bool asfile, bool force)
  {
    if (!force && DEBUG < 2)
      return;
    
    std::ostream* out = &std::cout;
    std::ofstream fs;
    if (name.size() > 0 && asfile)
    {
      fs.open(name, std::fstream::out | std::fstream::trunc);
      out = &fs;
    } else if (name.size() > 0 && !asfile)
      *out << name << ": ";

    *out << std::setprecision(precision);
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n-1; ++j)
        *out << A[n*i + j] << " ";
      *out << A[n*i + n-1] << std::endl;
    }
  }

  void
  dump_raw_matrix(const std::vector<double>& A, int m, int n, int precision,
                  const std::string& name, bool asfile, bool force)
  {
    dump_raw_matrix(A.data(), m, n, precision, name, asfile, force);
  }
}
