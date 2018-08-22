
/*******************************************************************************
 * Limiters
 *******************************************************************************/

/// Van Leer limiter. Used in TVD code
BoutReal VANLEER(BoutReal r) { return r + fabs(r) / (1.0 + fabs(r)); }

// Superbee limiter
BoutReal SUPERBEE(BoutReal r) {
  return BOUTMAX(0.0, BOUTMIN(2. * r, 1.0), BOUTMIN(r, 2.));
}

/*******************************************************************************
 * Basic derivative methods.
 * All expect to have an input grid cell at the same location as the output
 * Hence convert cell centred values -> centred values, or left -> left
 *******************************************************************************/

const BoutReal WENO_SMALL = 1.0e-8; // Small number for WENO schemes

////////////////////// FIRST DERIVATIVES /////////////////////

/// central, 2nd order
BoutReal DDX_C2(stencil &f) { return 0.5 * (f.p - f.m); }

/// central, 4th order
BoutReal DDX_C4(stencil &f) { return (8. * f.p - 8. * f.m + f.mm - f.pp) / 12.; }

/// Central WENO method, 2nd order (reverts to 1st order near shocks)
BoutReal DDX_CWENO2(stencil &f) {
  BoutReal isl, isr, isc;  // Smoothness indicators
  BoutReal al, ar, ac, sa; // Un-normalised weights
  BoutReal dl, dr, dc;     // Derivatives using different stencils

  dc = 0.5 * (f.p - f.m);
  dl = f.c - f.m;
  dr = f.p - f.c;

  isl = SQ(dl);
  isr = SQ(dr);
  isc = (13. / 3.) * SQ(f.p - 2. * f.c + f.m) + 0.25 * SQ(f.p - f.m);

  al = 0.25 / SQ(WENO_SMALL + isl);
  ar = 0.25 / SQ(WENO_SMALL + isr);
  ac = 0.5 / SQ(WENO_SMALL + isc);
  sa = al + ar + ac;

  return (al * dl + ar * dr + ac * dc) / sa;
}

// Smoothing 2nd order derivative
BoutReal DDX_S2(stencil &f) {

  // 4th-order differencing
  BoutReal result = (8. * f.p - 8. * f.m + f.mm - f.pp) / 12.;

  result += SIGN(f.c) * (f.pp - 4. * f.p + 6. * f.c - 4. * f.m + f.mm) / 12.;

  return result;
}

///////////////////// SECOND DERIVATIVES ////////////////////

/// Second derivative: Central, 2nd order
BoutReal D2DX2_C2(stencil &f) { return f.p + f.m - 2. * f.c; }

/// Second derivative: Central, 4th order
BoutReal D2DX2_C4(stencil &f) {
  return (-f.pp + 16. * f.p - 30. * f.c + 16. * f.m - f.mm) / 12.;
}

//////////////////////// UPWIND METHODS ///////////////////////

/// Upwinding: Central, 2nd order
BoutReal VDDX_C2(BoutReal vc, stencil &f) { return vc * 0.5 * (f.p - f.m); }

/// Upwinding: Central, 4th order
BoutReal VDDX_C4(BoutReal vc, stencil &f) {
  return vc * (8. * f.p - 8. * f.m + f.mm - f.pp) / 12.;
}

/// upwind, 1st order
BoutReal VDDX_U1(BoutReal vc, stencil &f) {
  return vc >= 0.0 ? vc * (f.c - f.m) : vc * (f.p - f.c);
}

/// upwind, 2nd order
BoutReal VDDX_U2(BoutReal vc, stencil &f) {
  return vc >= 0.0 ? vc * (1.5 * f.c - 2.0 * f.m + 0.5 * f.mm)
                   : vc * (-0.5 * f.pp + 2.0 * f.p - 1.5 * f.c);
}

/// upwind, 3rd order
BoutReal VDDX_U3(BoutReal vc, stencil &f) {
  return vc >= 0.0 ? vc*(4.*f.p - 12.*f.m + 2.*f.mm + 6.*f.c)/12.
    : vc*(-4.*f.m + 12.*f.p - 2.*f.pp - 6.*f.c)/12.;
}

/// 3rd-order WENO scheme
BoutReal VDDX_WENO3(BoutReal vc, stencil &f) {
  BoutReal deriv, w, r;

  if (vc > 0.0) {
    // Left-biased stencil

    r = (WENO_SMALL + SQ(f.c - 2.0 * f.m + f.mm)) /
        (WENO_SMALL + SQ(f.p - 2.0 * f.c + f.m));
    w = 1.0 / (1.0 + 2.0 * r * r);

    deriv = 0.5 * (f.p - f.m) - 0.5 * w * (-f.mm + 3. * f.m - 3. * f.c + f.p);

  } else {
    // Right-biased

    r = (WENO_SMALL + SQ(f.pp - 2.0 * f.p + f.c)) /
        (WENO_SMALL + SQ(f.p - 2.0 * f.c + f.m));
    w = 1.0 / (1.0 + 2.0 * r * r);

    deriv = 0.5 * (f.p - f.m) - 0.5 * w * (-f.m + 3. * f.c - 3. * f.p + f.pp);
  }

  return vc * deriv;
}

/// 3rd-order CWENO. Uses the upwinding code and split flux
BoutReal DDX_CWENO3(stencil &f) {
  BoutReal a, ma = fabs(f.c);
  // Split flux
  a = fabs(f.m);
  if (a > ma)
    ma = a;
  a = fabs(f.p);
  if (a > ma)
    ma = a;
  a = fabs(f.mm);
  if (a > ma)
    ma = a;
  a = fabs(f.pp);
  if (a > ma)
    ma = a;

  stencil sp, sm;

  sp.mm = f.mm + ma;
  sp.m = f.m + ma;
  sp.c = f.c + ma;
  sp.p = f.p + ma;
  sp.pp = f.pp + ma;
  sm.mm = ma - f.mm;
  sm.m = ma - f.m;
  sm.c = ma - f.c;
  sm.p = ma - f.p;
  sm.pp = ma - f.pp;

  return VDDX_WENO3(0.5, sp) + VDDX_WENO3(-0.5, sm);
}

//////////////////////// FLUX METHODS ///////////////////////

BoutReal FDDX_U1(stencil &v, stencil &f) {
  // Velocity at lower end
  BoutReal vs = 0.5 * (v.m + v.c);
  BoutReal result = (vs >= 0.0) ? vs * f.m : vs * f.c;
  // and at upper
  vs = 0.5 * (v.c + v.p);
  result -= (vs >= 0.0) ? vs * f.c : vs * f.p;

  return - result;
}

BoutReal FDDX_C2(stencil &v, stencil &f) { return 0.5 * (v.p * f.p - v.m * f.m); }

BoutReal FDDX_C4(stencil &v, stencil &f) {
  return (8. * v.p * f.p - 8. * v.m * f.m + v.mm * f.mm - v.pp * f.pp) / 12.;
}

//////////////////////// MUSCL scheme ///////////////////////

void DDX_KT_LR(const stencil &f, BoutReal &fLp, BoutReal &fRp, BoutReal &fLm,
               BoutReal &fRm) {
  // Limiter functions
  BoutReal phi = SUPERBEE((f.c - f.m) / (f.p - f.c));
  BoutReal phi_m = SUPERBEE((f.m - f.mm) / (f.c - f.m));
  BoutReal phi_p = SUPERBEE((f.p - f.c) / (f.pp - f.p));

  fLp = f.c + 0.5 * phi * (f.p - f.c);
  fRp = f.p - 0.5 * phi_p * (f.pp - f.p);

  fLm = f.m + 0.5 * phi_m * (f.c - f.m);
  fRm = f.c - 0.5 * phi * (f.p - f.c);
}

// du/dt = d/dx(f)  with maximum local velocity Vmax
BoutReal DDX_KT(const stencil &f, const stencil &u, const BoutReal Vmax) {
  BoutReal uLp, uRp, uLm, uRm;
  BoutReal fLp, fRp, fLm, fRm;

  DDX_KT_LR(u, uLp, uRp, uLm, uRm);
  DDX_KT_LR(f, fLp, fRp, fLm, fRm);

  BoutReal Fm = 0.5 * (fRm + fLm - Vmax * (uRm - uLm));
  BoutReal Fp = 0.5 * (fRp + fLp - Vmax * (uRp - uLp));

  return Fm - Fp;
}

/*******************************************************************************
 * Staggered differencing methods
 * These expect the output grid cell to be at a different location to the input
 *
 * The stencil no longer has a value in 'C' (centre)
 * instead, points are shifted as follows:
 *
 * mm  -> -3/2 h
 * m   -> -1/2 h
 * p   -> +1/2 h
 * pp  -? +3/2 h
 *
 * NOTE: Cell widths (dx, dy, dz) are currently defined as centre->centre
 * for the methods above. This is currently not taken account of, so large
 * variations in cell size will cause issues.
 *******************************************************************************/

/////////////////////// FIRST DERIVATIVES //////////////////////
// Map Centre -> Low or Low -> Centre

// Second order differencing (staggered)
BoutReal DDX_C2_stag(stencil &f) { return f.p - f.m; }

BoutReal DDX_C4_stag(stencil &f) { return (27. * (f.p - f.m) - (f.pp - f.mm)) / 24.; }

BoutReal D2DX2_C2_stag(stencil &f) { return (f.pp + f.mm - f.p - f.m) / 2.; }
/////////////////////////// UPWINDING ///////////////////////////
// Map (Low, Centre) -> Centre  or (Centre, Low) -> Low
// Hence v contains only (mm, m, p, pp) fields whilst f has 'c' too
//
// v.p is v at +1/2, v.m is at -1/2

BoutReal VDDX_U1_stag(stencil &v, stencil &f) {
  // Lower cell boundary
  BoutReal result = (v.m >= 0) ? v.m * f.m : v.m * f.c;

  // Upper cell boundary
  result -= (v.p >= 0) ? v.p * f.c : v.p * f.p;

  result *= -1;

  // result is now d/dx(v*f), but want v*d/dx(f) so subtract f*d/dx(v)
  result -= f.c * (v.p - v.m);

  return result;
}

BoutReal VDDX_U2_stag(stencil &v, stencil &f) {
  // Calculate d(v*f)/dx = (v*f)[i+1/2] - (v*f)[i-1/2]

  // Upper cell boundary
  BoutReal result = (v.p >= 0.) ? v.p * (1.5*f.c - 0.5*f.m) : v.p * (1.5*f.p - 0.5*f.pp);

  // Lower cell boundary
  result -= (v.m >= 0.) ? v.m * (1.5*f.m - 0.5*f.mm) : v.m * (1.5*f.c - 0.5*f.p);

  // result is now d/dx(v*f), but want v*d/dx(f) so subtract f*d/dx(v)
  result -= f.c * (v.p - v.m);

  return result;
}

BoutReal VDDX_C2_stag(stencil &v, stencil &f) {
  // Result is needed at location of f: interpolate v to f's location and take an
  // unstaggered derivative of f
  return 0.5 * (v.p + v.m) * 0.5 * (f.p - f.m);
}

BoutReal VDDX_C4_stag(stencil &v, stencil &f) {
  // Result is needed at location of f: interpolate v to f's location and take an
  // unstaggered derivative of f
  return (9. * (v.m + v.p) - v.mm - v.pp) / 16. * (8. * f.p - 8. * f.m + f.mm - f.pp) /
         12.;
}

/////////////////////////// FLUX ///////////////////////////
// Map (Low, Centre) -> Centre  or (Centre, Low) -> Low
// Hence v contains only (mm, m, p, pp) fields whilst f has 'c' too
//
// v.p is v at +1/2, v.m is at -1/2

BoutReal FDDX_U1_stag(stencil &v, stencil &f) {
  // Lower cell boundary
  BoutReal result = (v.m >= 0) ? v.m * f.m : v.m * f.c;

  // Upper cell boundary
  result -= (v.p >= 0) ? v.p * f.c : v.p * f.p;

  return - result;
}

BoutReal D4DX4_C2(stencil &f) { return (f.pp - 4. * f.p + 6. * f.c - 4. * f.m + f.mm); }


/*******************************************************************************
 * Lookup tables of functions. Map between names, codes and functions
 *******************************************************************************/

/// Translate between DIFF_METHOD codes, and functions
struct DiffLookup {
  DIFF_METHOD method;
  Mesh::deriv_func func;     // Single-argument differencing function
  Mesh::upwind_func up_func; // Upwinding function
  Mesh::flux_func fl_func;   // Flux function
  operator Mesh::deriv_func (){
    return func;
  }
  operator Mesh::upwind_func (){
    return up_func;
  }
  operator Mesh::flux_func (){
    return fl_func;
  }
};

/// Translate between short names, long names and DIFF_METHOD codes
struct DiffNameLookup {
  DIFF_METHOD method;
  const char *label; // Short name
  const char *name;  // Long name
};

/// Differential function name/code lookup
static DiffNameLookup DiffNameTable[] = {
    {DIFF_U1, "U1", "First order upwinding"},
    {DIFF_U2, "U2", "Second order upwinding"},
    {DIFF_C2, "C2", "Second order central"},
    {DIFF_W2, "W2", "Second order WENO"},
    {DIFF_W3, "W3", "Third order WENO"},
    {DIFF_C4, "C4", "Fourth order central"},
    {DIFF_U3, "U3", "Third order upwinding"},
    {DIFF_U3, "U4", "Third order upwinding (Can't do 4th order yet)."},
    {DIFF_S2, "S2", "Smoothing 2nd order"},
    {DIFF_FFT, "FFT", "FFT"},
    {DIFF_SPLIT, "SPLIT", "Split into upwind and central"},
    {DIFF_DEFAULT, nullptr, nullptr}}; // Use to terminate the list

/// First derivative lookup table
static DiffLookup FirstDerivTable[] = {
    {DIFF_C2, DDX_C2, nullptr, nullptr},     {DIFF_W2, DDX_CWENO2, nullptr, nullptr},
    {DIFF_W3, DDX_CWENO3, nullptr, nullptr}, {DIFF_C4, DDX_C4, nullptr, nullptr},
    {DIFF_S2, DDX_S2, nullptr, nullptr},     {DIFF_FFT, nullptr, nullptr, nullptr},
    {DIFF_DEFAULT, nullptr, nullptr, nullptr}};

/// Second derivative lookup table
static DiffLookup SecondDerivTable[] = {{DIFF_C2, D2DX2_C2, nullptr, nullptr},
                                        {DIFF_C4, D2DX2_C4, nullptr, nullptr},
                                        {DIFF_FFT, nullptr, nullptr, nullptr},
                                        {DIFF_DEFAULT, nullptr, nullptr, nullptr}};

/// Upwinding functions lookup table
static DiffLookup UpwindTable[] = {
    {DIFF_U1, nullptr, VDDX_U1, nullptr},    {DIFF_U2, nullptr, VDDX_U2, nullptr},
    {DIFF_C2, nullptr, VDDX_C2, nullptr},    {DIFF_U3, nullptr, VDDX_U3, nullptr},
    {DIFF_W3, nullptr, VDDX_WENO3, nullptr}, {DIFF_C4, nullptr, VDDX_C4, nullptr},
    {DIFF_DEFAULT, nullptr, nullptr, nullptr}};

/// Flux functions lookup table
static DiffLookup FluxTable[] = {
    {DIFF_SPLIT, nullptr, nullptr, nullptr},   {DIFF_U1, nullptr, nullptr, FDDX_U1},
    {DIFF_C2, nullptr, nullptr, FDDX_C2},   {DIFF_C4, nullptr, nullptr, FDDX_C4},
    {DIFF_DEFAULT, nullptr, nullptr, nullptr}};

/// First staggered derivative lookup
static DiffLookup FirstStagDerivTable[] = {{DIFF_C2, DDX_C2_stag, nullptr, nullptr},
                                           {DIFF_C4, DDX_C4_stag, nullptr, nullptr},
                                           {DIFF_DEFAULT, nullptr, nullptr, nullptr}};

/// Second staggered derivative lookup
static DiffLookup SecondStagDerivTable[] = {{DIFF_C2, D2DX2_C2_stag, nullptr, nullptr},
                                            {DIFF_DEFAULT, nullptr, nullptr, nullptr}};

/// Upwinding staggered lookup
static DiffLookup UpwindStagTable[] = {{DIFF_U1, nullptr, nullptr, VDDX_U1_stag},
                                       {DIFF_U2, nullptr, nullptr, VDDX_U2_stag},
                                       {DIFF_C2, nullptr, nullptr, VDDX_C2_stag},
                                       {DIFF_C4, nullptr, nullptr, VDDX_C4_stag},
                                       {DIFF_DEFAULT, nullptr, nullptr, nullptr}};

/// Flux staggered lookup
static DiffLookup FluxStagTable[] = {{DIFF_SPLIT, nullptr, nullptr, nullptr},
                                     {DIFF_U1, nullptr, nullptr, FDDX_U1_stag},
                                     {DIFF_DEFAULT, nullptr, nullptr, nullptr}};

/*******************************************************************************
 * Routines to use the above tables to map between function codes, names
 * and pointers
 *******************************************************************************/


/// Test if a given DIFF_METHOD exists in a table
bool isImplemented(DiffLookup *table, DIFF_METHOD method) {
  int i = 0;
  do {
    if (table[i].method == method)
      return true;
    i++;
  } while (table[i].method != DIFF_DEFAULT);

  return false;
}


DiffLookup lookupFunc(DiffLookup * table, DIFF_METHOD method) {
  int i=0;
  for (int i=0; ; ++i){
    if (table[i].method == method) {
      return table[i];
    }
    if (table[i].method == DIFF_DEFAULT){
      return table[i];
    }
  }
}

void printFuncName(DIFF_METHOD method) {
  // Find this entry
  int i = 0;
  do {
    if (DiffNameTable[i].method == method) {
      output_info.write(" %s (%s)\n", DiffNameTable[i].name, DiffNameTable[i].label);
      return;
    }
    i++;
  } while (DiffNameTable[i].method != DIFF_DEFAULT);

  // None
  output_error.write(" == INVALID DIFFERENTIAL METHOD ==\n");
}

/// This function is used during initialisation only (i.e. doesn't need to be particularly
/// fast) Returns DIFF_METHOD, rather than function so can be applied to central and
/// upwind tables
DiffLookup lookupFunc(DiffLookup *table, const std::string & label){

  // Loop through the name lookup table
  for (int i = 0; DiffNameTable[i].method != DIFF_DEFAULT; ++i) {
    if (strcasecmp(label.c_str(), DiffNameTable[i].label) == 0) { // Whole match
      auto method=DiffNameTable[i].method;
      if (isImplemented(table, method)) {
        printFuncName(method);
        for (int j=0;;++j){
          if (table[j].method == method){
            return table[j];
          }
        }
      }
    }
  }

  // No exact match, so throw
  std::string avail{};
  for (int i = 0; DiffNameTable[i].method != DIFF_DEFAULT; ++i) {
    avail += DiffNameTable[i].label;
    avail += "\n";
  }
  throw BoutException("Unknown option %s.\nAvailable options are:\n%s", label.c_str(),
                      avail.c_str());
}


/*******************************************************************************
 * Default functions
 *
 *
 *******************************************************************************/

// Central -> Central (or Left -> Left) functions
Mesh::deriv_func fDDX, fDDY, fDDZ;       ///< Differencing methods for each dimension
Mesh::deriv_func fD2DX2, fD2DY2, fD2DZ2; ///< second differential operators
Mesh::upwind_func fVDDX, fVDDY, fVDDZ;   ///< Upwind functions in the three directions
Mesh::flux_func fFDDX, fFDDY, fFDDZ;     ///< Default flux functions

// Central -> Left (or Left -> Central) functions
Mesh::deriv_func sfDDX, sfDDY, sfDDZ;
Mesh::deriv_func sfD2DX2, sfD2DY2, sfD2DZ2;
Mesh::flux_func sfVDDX, sfVDDY, sfVDDZ;
Mesh::flux_func sfFDDX, sfFDDY, sfFDDZ;

/*******************************************************************************
 * Initialisation
 *******************************************************************************/

/// Set the derivative method, given a table and option name
template <typename T, typename Ts>
void derivs_set(std::vector<Options *> options, DiffLookup *table, DiffLookup *stable,
                const std::string &name, const std::string &def, T &f, Ts &sf,
                bool staggerGrids) {
  TRACE("derivs_set()");
  output_info.write("\t%-12s: ", name.c_str());
  string label = def;
  for (auto &opts : options) {
    if (opts->isSet(name)) {
      opts->get(name, label, "");
      break;
    }
  }

  f = lookupFunc(table, label); // Find the function

  label = def;
  if (staggerGrids) {
    output_info.write("\tStag. %-6s: ", name.c_str());
    for (auto &_name : {name + "stag", name}) {
      for (auto &opts : options) {
        if (opts->isSet(_name)) {
          opts->get(_name, label, "");
          sf = lookupFunc(stable, label); // Find the function
          return;
        }
      }
    }
  }
  sf = lookupFunc(stable, label); // Find the function
}

/// Initialise derivatives from options
void derivs_initialise(Options *optionbase, std::string sec, bool staggerGrids,
                       Mesh::deriv_func &fdd, Mesh::deriv_func &sfdd,
                       Mesh::deriv_func &fd2d, Mesh::deriv_func &sfd2d,
                       Mesh::upwind_func &fu, Mesh::flux_func &sfu, Mesh::flux_func &ff,
                       Mesh::flux_func &sff) {
  std::vector<Options *> options = {optionbase->getSection(sec),
                                    optionbase->getSection("diff")};
  derivs_set(options, FirstDerivTable, FirstStagDerivTable, "First", "C2", fdd, sfdd,
             staggerGrids);

  derivs_set(options, SecondDerivTable, SecondStagDerivTable, "Second", "C2", fd2d, sfd2d,
             staggerGrids);

  derivs_set(options, UpwindTable, UpwindStagTable, "Upwind", "U1", fu, sfu,
             staggerGrids);

  derivs_set(options, FluxTable, FluxStagTable, "Flux", "U1", ff, sff, staggerGrids);
}

/// Initialise the derivative methods. Must be called before any derivatives are used
void Mesh::derivs_init(Options *options) {
  TRACE("Initialising derivatives");

  output_info.write("Setting X differencing methods\n");
  derivs_initialise(options, "ddx", StaggerGrids, fDDX, sfDDX, fD2DX2, sfD2DX2, fVDDX,
                    sfVDDX, fFDDX, sfFDDX);

  if ((fDDX == nullptr) || (fD2DX2 == nullptr))
    throw BoutException("FFT cannot be used in X\n");

  output_info.write("Setting Y differencing methods\n");
  derivs_initialise(options, "ddy", StaggerGrids, fDDY, sfDDY, fD2DY2, sfD2DY2, fVDDY,
                    sfVDDY, fFDDY, sfFDDY);

  if ((fDDY == nullptr) || (fD2DY2 == nullptr))
    throw BoutException("FFT cannot be used in Y\n");

  output_info.write("Setting Z differencing methods\n");
  derivs_initialise(options, "ddz", StaggerGrids, fDDZ, sfDDZ, fD2DZ2, sfD2DZ2, fVDDZ,
                    sfVDDZ, fFDDZ, sfFDDZ);

  // Get the fraction of modes filtered out in FFT derivatives
  options->getSection("ddz")->get("fft_filter", fft_derivs_filter, 0.0);
}

/////////////////////////////////////////////
// Routine to populate stencils
/////////////////////////////////////////////

void NG_2_DIR_X ( stencil &s, const Field3D& var, const Ind3D& i){
  s.mm = var[i.xmm()];
  s.m = var[i.xm()];
  s.c = var[i];
  s.p = var[i.xp()];
  s.pp = var[i.xpp()];
}

void NG_1_DIR_X ( stencil &s, const Field3D& var, const Ind3D& i){
  s.m = var[i.xm()];
  s.c = var[i];
  s.p = var[i.xp()];
}

void NG_2_DIR_X ( stencil &s, const Field2D& var, const Ind2D& i){
  s.mm = var[i.xmm()];
  s.m = var[i.xm()];
  s.c = var[i];
  s.p = var[i.xp()];
  s.pp = var[i.xpp()];
}

void NG_1_DIR_X ( stencil &s, const Field2D& var, const Ind2D& i){
  s.m = var[i.xm()];
  s.c = var[i];
  s.p = var[i.xp()];
}
