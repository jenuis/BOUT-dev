#include <globals.hxx>
#include <boundary_factory.hxx>
#include <boundary_standard.hxx>
#include <utils.hxx>

#include <list>
#include <string>
using std::list;
using std::string;

#include <output.hxx>

BoundaryFactory *BoundaryFactory::instance = nullptr;

BoundaryFactory::BoundaryFactory() {
  add(new BoundaryDirichlet(), "dirichlet");
  add(new BoundaryDirichlet(), "dirichlet_o2"); // Synonym for "dirichlet"
  add(new BoundaryDirichlet_2ndOrder(), "dirichlet_2ndorder"); // Deprecated
  add(new BoundaryDirichlet_O3(), "dirichlet_o3");
  add(new BoundaryDirichlet_O4(), "dirichlet_o4");
  add(new BoundaryDirichlet_4thOrder(), "dirichlet_4thorder");
  add(new BoundaryNeumann(), "neumann");
  add(new BoundaryNeumann(), "neumann_O2"); // Synonym for "neumann"
  add(new BoundaryNeumann2(), "neumann2"); // Deprecated
  add(new BoundaryNeumann_2ndOrder(), "neumann_2ndorder"); // Deprecated
  add(new BoundaryNeumann_4thOrder(), "neumann_4thorder");
  add(new BoundaryNeumann_O4(), "neumann_O4");
  add(new BoundaryNeumannPar(), "neumannpar");
  add(new BoundaryNeumann_NonOrthogonal(), "neumann_nonorthogonal");
  add(new BoundaryRobin(), "robin");
  add(new BoundaryConstGradient(), "constgradient");
  add(new BoundaryZeroLaplace(), "zerolaplace");
  add(new BoundaryZeroLaplace2(), "zerolaplace2");
  add(new BoundaryConstLaplace(), "constlaplace");
  add(new BoundaryFree(), "free");
  add(new BoundaryFree_O2(), "free_o2");
  add(new BoundaryFree_O3(), "free_o3");
  
  addMod(new BoundaryRelax(), "relax");
  addMod(new BoundaryWidth(), "width");
  addMod(new BoundaryToFieldAligned(), "toFieldAligned");
  addMod(new BoundaryFromFieldAligned(), "fromFieldAligned");

  // Parallel boundaries
  add(new BoundaryOpPar_dirichlet(), "parallel_dirichlet");
  add(new BoundaryOpPar_dirichlet_O3(), "parallel_dirichlet_O3");
  add(new BoundaryOpPar_dirichlet_interp(), "parallel_dirichlet_interp");
  add(new BoundaryOpPar_neumann(), "parallel_neumann");
}

BoundaryFactory::~BoundaryFactory() {
  // Free any boundaries
  for (const auto &it : opmap) {
    delete it.second;
  }
  for (const auto &it : modmap) {
    delete it.second;
  }
  for (const auto &it : par_opmap) {
    delete it.second;
  }
}

BoundaryFactory* BoundaryFactory::getInstance() {
  if (instance == nullptr) {
    // Create the singleton object
    instance = new BoundaryFactory();
  }
  return instance;
}

void BoundaryFactory::cleanup() {
  if (instance == nullptr)
    return;

  // Just delete the instance
  delete instance;
  instance = nullptr;
}

namespace {
  template<typename T>
  using OpRegion = typename std::conditional<std::is_same<T, BoundaryRegionPar>::value, BoundaryRegionPar, BoundaryRegion>::type;
}

template<typename T>
T* BoundaryFactory::create(const string &name, OpRegion<T> *region) {

  // Search for a string of the form: modifier(operation)
  auto pos = name.find('(');
  if(pos == string::npos) {
    // No more (opening) brackets. Should be a boundary operation
    // Need to strip whitespace

    if( (name == "null") || (name == "none") )
      return nullptr;

    T *op;
    if (std::is_same<T, BoundaryOpPar>::value) {
      op = findBoundaryOpPar(trim(name));
    } else {
      op = findBoundaryOp(trim(name));
    }
    if (op == nullptr) {
      throw BoutException("Could not find boundary condition '%s'",  name.c_str());
    }

    // Clone the boundary operation, passing the region to operate over and an empty args list
    list<string> args;
    return op->clone(OpRegion<T>* region, args);
  }

  // Contains a bracket. Find the last bracket and remove
  auto pos2 = name.rfind(')');
  if(pos2 == string::npos) {
    output_warn << "\tWARNING: Unmatched brackets in boundary condition: " << name << endl;
  }

  // Find the function name before the bracket
  string func = trim(name.substr(0,pos));
  // And the argument inside the bracket
  string arg = trim(name.substr(pos+1, pos2-pos-1));
  // Split the argument on commas
  // NOTE: Commas could be part of sub-expressions, so
  //       need to take account of brackets
  list<string> arglist;
  int level = 0;
  int start = 0;
  for(string::size_type i = 0;i<arg.length();i++) {
    switch(arg[i]) {
    case '(':
    case '[':
    case '<':
      level++;
      break;
    case ')':
    case ']':
    case '>':
      level--;
      break;
    case ',': {
      if(level == 0) {
        string s = arg.substr(start, i);
        arglist.push_back(trim(s));
        start = i+1;
      }
      break;
    }
    };
  }
  string s = arg.substr(start, arg.length());
  arglist.push_back(trim(s));

  /*
    list<string> arglist = strsplit(arg, ',');
    for(list<string>::iterator it=arglist.begin(); it != arglist.end(); it++) {
    // Trim each argument
    (*it) = trim(*it);
    }
  */

  if (std::is_same<T, BoundaryOp>::value) {
    // Test if func is a modifier
    BoundaryModifier *mod = findBoundaryMod(func);
    if (mod != nullptr) {
      // The first argument should be an operation
      T *op = create(arglist.front(), region);
      if (op == nullptr)
        return nullptr;

      // Remove the first element (name of operation)
      arglist.pop_front();

      // Clone the modifier, passing in the operator and remaining strings as argument
      return mod->cloneMod(op, arglist);
    }

    T *op = findBoundaryOp(trim(func));
    if (op != nullptr) {
      // An operation with arguments
      return op->clone(region, arglist);
    }
  } else {
    // Parallel boundary
    T *pop = findBoundaryOpPar(trim(func));
    if (pop != nullptr) {
      // An operation with arguments
      return pop->clone(region, arglist);
    }
  }

  // Otherwise nothing matches
  throw BoutException("  Boundary setting is neither an operation nor modifier: %s\n",func.c_str());

  return nullptr;
}

BoundaryOp* BoundaryFactory::create(const char* name, BoundaryRegion*region) {
  return create(string(name), region);
}

BoundaryOpPar* BoundaryFactory::create(const char* name, BoundaryRegionPar *region) {
  return create(string(name), region);
}

template<typename T>
T* BoundaryFactory::createFromOptions(const string &varname, OpRegion<T>* region) {
  if (region == nullptr)
    return nullptr;

  output_info << "\t" << region->label << " region: ";

  string prefix("bndry_");

  string side;
  switch(region->location) {
  case BNDRY_XIN: {
    side = "xin";
    break;
  }
  case BNDRY_XOUT: {
    side = "xout";
    break;
  }
  case BNDRY_YDOWN: {
    side = "ydown";
    break;
  }
  case BNDRY_YUP: {
    side = "yup";
    break;
  }
  case BNDRY_PAR_FWD: {
    side = "par_yup";
    break;
  }
  case BNDRY_PAR_BKWD: {
    side = "par_ydown";
    break;
  }
  default: {
    side = "all";
    break;
  }
  }

  // Get options
  Options *options = Options::getRoot();

  // Get variable options
  Options *varOpts = options->getSection(varname);
  string set;

  /// First try looking for (var, region)
  if(varOpts->isSet(prefix+region->label)) {
    varOpts->get(prefix+region->label, set, "");
    return create(set, region);
  }

  /// Then (var, side)
  if(varOpts->isSet(prefix+side)) {
    varOpts->get(prefix+side, set, "");
    return create(set, region);
  }

  /// Then (var, all)
  if(std::is_same<T, BoundaryOpPar>::value) {
    if(varOpts->isSet(prefix+"par_all")) {
      varOpts->get(prefix+"par_all", set, "");
      return create(set, region);
    }
  } else {
    if(varOpts->isSet(prefix+"all")) {
      varOpts->get(prefix+"all", set, "");
      return create(set, region);
    }
  }

  // Get the "all" options
  varOpts = options->getSection("All");

  /// Then (all, region)
  if(varOpts->isSet(prefix+region->label)) {
    varOpts->get(prefix+region->label, set, "");
    return create(set, region);
  }

  /// Then (all, side)
  if(varOpts->isSet(prefix+side)) {
    varOpts->get(prefix+side, set, "");
    return create(set, region);
  }

  /// Then (all, all)
  if(std::is_same<T, BoundaryOpPar>::value) {
    // Different default for parallel boundary regions
    varOpts->get(prefix+"par_all", set, "parallel_dirichlet");
  } else {
    varOpts->get(prefix+"all", set, "dirichlet");
  }
  return create(set, region);
  // Defaults to Dirichlet conditions, to prevent undefined boundary
  // values. If a user want to override, specify "none" or "null"
}

BoundaryOp* BoundaryFactory::createFromOptions(const char* varname, BoundaryRegion* region) {
  return createFromOptions(string(varname), region);
}

BoundaryOpPar* BoundaryFactory::createFromOptions(const char* varname, BoundaryRegionPar* region) {
  return createFromOptionsPar(string(varname), region);
}

void BoundaryFactory::add(BoundaryOp* bop, const string &name) {
  if ((findBoundaryMod(name) != nullptr) || (findBoundaryOp(name) != nullptr)) {
    // error - already exists
    output_error << "ERROR: Trying to add an already existing boundary: " << name << endl;
    return;
  }
  opmap[lowercase(name)] = bop;
}

void BoundaryFactory::add(BoundaryOp* bop, const char *name) {
  add(bop, string(name));
}

void BoundaryFactory::add(BoundaryOpPar* bop, const string &name) {
  if (findBoundaryOpPar(name) != nullptr) {
    // error - already exists
    output_error << "ERROR: Trying to add an already existing boundary: " << name << endl;
    return;
  }
  par_opmap[lowercase(name)] = bop;
}

void BoundaryFactory::add(BoundaryOpPar* bop, const char *name) {
  add(bop, string(name));
}

void BoundaryFactory::addMod(BoundaryModifier* bmod, const string &name) {
  if ((findBoundaryMod(name) != nullptr) || (findBoundaryOp(name) != nullptr)) {
    // error - already exists
    output_error << "ERROR: Trying to add an already existing boundary modifier: " << name << endl;
    return;
  }
  modmap[lowercase(name)] = bmod;
}

void BoundaryFactory::addMod(BoundaryModifier* bmod, const char *name) {
  addMod(bmod, string(name));
}

BoundaryOp* BoundaryFactory::findBoundaryOp(const string &s) {
  map<string,BoundaryOp*>::iterator it;
  it = opmap.find(lowercase(s));
  if(it == opmap.end())
    return nullptr;
  return it->second;
}

BoundaryModifier* BoundaryFactory::findBoundaryMod(const string &s) {
  map<string,BoundaryModifier*>::iterator it;
  it = modmap.find(lowercase(s));
  if(it == modmap.end())
    return nullptr;
  return it->second;
}

BoundaryOpPar* BoundaryFactory::findBoundaryOpPar(const string &s) {
  map<string,BoundaryOpPar*>::iterator it;
  it = par_opmap.find(lowercase(s));
  if(it == par_opmap.end())
    return nullptr;
  return it->second;
}
