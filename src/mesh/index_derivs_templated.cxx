/**************************************************************************
 * Basic derivative methods in mesh index space
 *
 *
 * Four kinds of differencing methods:
 *
 * 1. First derivative DD*
 *    Central differencing e.g. Div(f)
 *
 * 2. Second derivatives D2D*2
 *    Central differencing e.g. Delp2(f)
 *
 * 3. Upwinding VDD*
 *    Terms like v*Grad(f)
 *
 * 4. Flux methods FDD* (e.g. flux conserving, limiting)
 *    Div(v*f)
 *
 * Changelog
 * =========
 *
 * 2014-11-22   Ben Dudson  <benjamin.dudson@york.ac.uk>
 *    o Moved here from sys/derivs, made part of Mesh
 *
 **************************************************************************
 * Copyright 2010 B.D.Dudson, S.Farley, M.V.Umansky, X.Q.Xu
 *
 * Contact: Ben Dudson, bd512@york.ac.uk
 *
 * This file is part of BOUT++.
 *
 * BOUT++ is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * BOUT++ is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with BOUT++.  If not, see <http://www.gnu.org/licenses/>.
 *
 **************************************************************************/

#include <bout/constants.hxx>
#include <derivs.hxx>
#include <fft.hxx>
#include <globals.hxx>
#include <interpolation.hxx>
#include <bout/constants.hxx>
#include <bout/openmpwrap.hxx>
#include <bout/region.hxx>
#include <msg_stack.hxx>
#include <stencils.hxx>
#include <utils.hxx>
#include <unused.hxx>
#include <bout/openmpwrap.hxx>
#include <cmath>
#include <stdlib.h>
#include <string.h>

#include <output.hxx>

#include <bout/mesh.hxx>

#include "methods.hxx"
/*******************************************************************************
 * Apply differential operators. These are fairly brain-dead functions
 * which apply a derivative function to a field (sort of like map). Decisions
 * of what to apply are made in the DDX,DDY and DDZ functions lower down.
 *
 * loc  is the cell location of the result
 *******************************************************************************/

// X derivative
template<Mesh::deriv_func func, Mesh::stencil_2d get_vals>
const Field2D Mesh::applyXdiff(const Field2D &var,
                               CELL_LOC loc, REGION region) {
  ASSERT1(this == var.getMesh());
  ASSERT1(var.isAllocated());

  if (var.getNx() == 1) {
    return Field2D(0., this);
  }

  CELL_LOC diffloc = var.getLocation();

  Field2D result(this);
  result.allocate(); // Make sure data allocated

  if (this->StaggerGrids && (loc != CELL_DEFAULT) && (loc != var.getLocation())) {
    // Staggered differencing

    CELL_LOC location = var.getLocation();

    if (this->xstart > 1) {
      // More than one guard cell, so set pp and mm values
      // This allows higher-order methods to be used
      stencil s;
      for (const auto &i : result.region(region)) {
        s.c = var[i];
        s.p = var[i.xp()];
        s.m = var[i.xm()];
        s.pp = var[i.offset(2, 0, 0)];
        s.mm = var[i.offset(-2, 0, 0)];

        if ((location == CELL_CENTRE) && (loc == CELL_XLOW)) {
          // Producing a stencil centred around a lower X value
          s.pp = s.p;
          s.p = s.c;
        } else if (location == CELL_XLOW) {
          // Stencil centred around a cell centre
          s.mm = s.m;
          s.m = s.c;
        }

        result[i] = func(s);
      }
    } else {
      // Only one guard cell, so no pp or mm values
      for (const auto &i : result.region(region)) {
        stencil s;
        s.c = var[i];
        s.p = var[i.xp()];
        s.m = var[i.xm()];
        s.pp = nan("");
        s.mm = nan("");

        if ((location == CELL_CENTRE) && (loc == CELL_XLOW)) {
          // Producing a stencil centred around a lower X value
          s.pp = s.p;
          s.p = s.c;
        } else if (location == CELL_XLOW) {
          // Stencil centred around a cell centre
          s.mm = s.m;
          s.m = s.c;
        }

        result[i] = func(s);
      }
    }

  } else {
    // Non-staggered differencing
    
    BOUT_OMP(parallel)
    {
      stencil s;
      BLOCK_REGION_LOOP_PARALLEL_SECTION(mesh->getRegion2D("RGN_NOBNDRY"), i,
					 // s.mm = var[i.xmm()];
					 // s.m = var[i.xm()];
					 // s.c = var[i];
					 // s.p = var[i.xp()];
					 // s.pp = var[i.xpp()];
					 get_vals(s, var, i);
					 result[i] = func(s);
					 )
	}
  }

  result.setLocation(diffloc);

#if CHECK > 0
  // Mark boundaries as invalid
  result.bndry_xin = result.bndry_xout = result.bndry_yup = result.bndry_ydown = false;
#endif

  return result;
}

template<Mesh::deriv_func func, Mesh::stencil_3d get_vals>
const Field3D Mesh::applyXdiff(const Field3D &var, 
                               CELL_LOC loc, REGION region) {
  // Check that the mesh is correct
  ASSERT1(this == var.getMesh());
  // Check that the input variable has data
  ASSERT1(var.isAllocated());

  if (var.getNx() == 1) {
    return Field3D(0., this);
  }

  CELL_LOC diffloc = var.getLocation();

  Field3D result(this);
  result.allocate(); // Make sure data allocated

  if (this->StaggerGrids && (loc != CELL_DEFAULT) && (loc != var.getLocation())) {
    // Staggered differencing

    CELL_LOC location = var.getLocation();

    if (this->xstart > 1) {
      // More than one guard cell, so set pp and mm values
      // This allows higher-order methods to be used
      stencil s;
      for (const auto &i : result.region(region)) {
        s.c = var[i];
        s.p = var[i.xp()];
        s.m = var[i.xm()];
        s.pp = var[i.offset(2, 0, 0)];
        s.mm = var[i.offset(-2, 0, 0)];

        if ((location == CELL_CENTRE) && (loc == CELL_XLOW)) {
          // Producing a stencil centred around a lower X value
          s.pp = s.p;
          s.p = s.c;
        } else if (location == CELL_XLOW) {
          // Stencil centred around a cell centre
          s.mm = s.m;
          s.m = s.c;
        }

        result[i] = func(s);
      }
    } else {
      // Only one guard cell, so no pp or mm values
      stencil s;
      s.pp = nan("");
      s.mm = nan("");
      for (const auto &i : result.region(region)) {
        s.c = var[i];
        s.p = var[i.xp()];
        s.m = var[i.xm()];

        if ((location == CELL_CENTRE) && (loc == CELL_XLOW)) {
          // Producing a stencil centred around a lower X value
          s.pp = s.p;
          s.p = s.c;
        } else if (location == CELL_XLOW) {
          // Stencil centred around a cell centre
          s.mm = s.m;
          s.m = s.c;
        }

        result[i] = func(s);
      }
    }

  } else {
    // Non-staggered differencing

    BOUT_OMP(parallel)
    {
      stencil s;
      //for (const auto &i : result.region(region)) {
      BLOCK_REGION_LOOP_PARALLEL_SECTION(mesh->getRegion3D("RGN_NOBNDRY"), i,
					 get_vals(s, var, i);
					 result[i] = func(s);
					 )
	}
  }


  result.setLocation(diffloc);

#if CHECK > 0
  // Mark boundaries as invalid
  result.bndry_xin = result.bndry_xout = result.bndry_yup = result.bndry_ydown = false;
#endif

  return result;
}

// Y derivative

const Field2D Mesh::applyYdiff(const Field2D &var, Mesh::deriv_func func, CELL_LOC UNUSED(loc),
                               REGION region) {
  ASSERT1(this == var.getMesh());
  // Check that the input variable has data
  ASSERT1(var.isAllocated());

  if (var.getNy() == 1) {
    return Field2D(0., this);
  }

  CELL_LOC diffloc = var.getLocation();

  Field2D result(this);
  result.allocate(); // Make sure data allocated

  if (this->ystart > 1) {
    // More than one guard cell, so set pp and mm values
    // This allows higher-order methods to be used

    for (const auto &i : result.region(region)) {
      // Set stencils
      stencil s;
      s.c = var[i];
      s.p = var[i.yp()];
      s.m = var[i.ym()];
      s.pp = var[i.offset(0, 2, 0)];
      s.mm = var[i.offset(0, -2, 0)];

      result[i] = func(s);
    }
  } else {
    // Only one guard cell, so no pp or mm values
    for (const auto &i : result.region(region)) {
      // Set stencils
      stencil s;
      s.c = var[i];
      s.p = var[i.yp()];
      s.m = var[i.ym()];
      s.pp = nan("");
      s.mm = nan("");

      result[i] = func(s);
    }
  }

  result.setLocation(diffloc);

#if CHECK > 0
  // Mark boundaries as invalid
  result.bndry_yup = result.bndry_ydown = false;
#endif

  return result;
}

const Field3D Mesh::applyYdiff(const Field3D &var, Mesh::deriv_func func, CELL_LOC loc,
                               REGION region) {
  ASSERT1(this == var.getMesh());
  // Check that the input variable has data
  ASSERT1(var.isAllocated());

  if (var.getNy() == 1) {
    return Field3D(0., this);
  }

  CELL_LOC diffloc = var.getLocation();

  Field3D result(this);
  result.allocate(); // Make sure data allocated

  if (var.hasYupYdown() && ((&var.yup() != &var) || (&var.ydown() != &var))) {
    // Field "var" has distinct yup and ydown fields which
    // will be used to calculate a derivative along
    // the magnetic field

    if (this->StaggerGrids && (loc != CELL_DEFAULT) && (loc != var.getLocation())) {
      // Staggered differencing

      // Cell location of the input field
      CELL_LOC location = var.getLocation();

      stencil s;
      s.pp = nan("");
      s.mm = nan("");
      for (const auto &i : result.region(region)) {
        // Set stencils
        s.c = var[i];
        s.p = var.yup()[i.yp()];
        s.m = var.ydown()[i.ym()];

        if ((location == CELL_CENTRE) && (loc == CELL_YLOW)) {
          // Producing a stencil centred around a lower Y value
          s.pp = s.p;
          s.p = s.c;
        } else if (location == CELL_YLOW) {
          // Stencil centred around a cell centre
          s.mm = s.m;
          s.m = s.c;
        }

        result[i] = func(s);
      }
    } else {
      // Non-staggered
      stencil s;
      s.pp = nan("");
      s.mm = nan("");
      for (const auto &i : result.region(region)) {
        // Set stencils
        s.c = var[i];
        s.p = var.yup()[i.yp()];
        s.m = var.ydown()[i.ym()];

        result[i] = func(s);
      }
    }
  } else {
    // var has no yup/ydown fields, so we need to shift into field-aligned coordinates

    Field3D var_fa = this->toFieldAligned(var);

    if (this->StaggerGrids && (loc != CELL_DEFAULT) && (loc != var.getLocation())) {
      // Staggered differencing

      // Cell location of the input field
      CELL_LOC location = var.getLocation();

      if (this->ystart > 1) {
        // More than one guard cell, so set pp and mm values
        // This allows higher-order methods to be used
        stencil s;
        for (const auto &i : result.region(region)) {
          // Set stencils
          s.c = var_fa[i];
          s.p = var_fa[i.yp()];
          s.m = var_fa[i.ym()];
          s.pp = var_fa[i.offset(0, 2, 0)];
          s.mm = var_fa[i.offset(0, -2, 0)];

          if ((location == CELL_CENTRE) && (loc == CELL_YLOW)) {
            // Producing a stencil centred around a lower Y value
            s.pp = s.p;
            s.p = s.c;
          } else if (location == CELL_YLOW) {
            // Stencil centred around a cell centre
            s.mm = s.m;
            s.m = s.c;
          }

          result[i] = func(s);
        }
      } else {
        // Only one guard cell, so no pp or mm values
        stencil s;
        s.pp = nan("");
        s.mm = nan("");
        for (const auto &i : result.region(region)) {
          // Set stencils
          s.c = var_fa[i];
          s.p = var_fa[i.yp()];
          s.m = var_fa[i.ym()];

          if ((location == CELL_CENTRE) && (loc == CELL_YLOW)) {
            // Producing a stencil centred around a lower Y value
            s.pp = s.p;
            s.p = s.c;
          } else if (location == CELL_YLOW) {
            // Stencil centred around a cell centre
            s.mm = s.m;
            s.m = s.c;
          }

          result[i] = func(s);
        }
      }

    } else {
      // Non-staggered differencing

      if (this->ystart > 1) {
        // More than one guard cell, so set pp and mm values
        // This allows higher-order methods to be used
        stencil s;
        for (const auto &i : result.region(region)) {
          // Set stencils
          s.c = var_fa[i];
          s.p = var_fa[i.yp()];
          s.m = var_fa[i.ym()];
          s.pp = var_fa[i.offset(0, 2, 0)];
          s.mm = var_fa[i.offset(0, -2, 0)];

          result[i] = func(s);
        }
      } else {
        // Only one guard cell, so no pp or mm values
        stencil s;
        s.pp = nan("");
        s.mm = nan("");
        for (const auto &i : result.region(region)) {
          // Set stencils
          s.c = var_fa[i];
          s.p = var_fa[i.yp()];
          s.m = var_fa[i.ym()];

          result[i] = func(s);
        }
      }
    }

    // Shift result back

    result = this->fromFieldAligned(result);
  }

  result.setLocation(diffloc);

#if CHECK > 0
  // Mark boundaries as invalid
  result.bndry_xin = result.bndry_xout = result.bndry_yup = result.bndry_ydown = false;
#endif

  return result;
}

// Z derivative

const Field3D Mesh::applyZdiff(const Field3D &var, Mesh::deriv_func func, CELL_LOC loc,
                               REGION region) {
  ASSERT1(this == var.getMesh());
  // Check that the input variable has data
  ASSERT1(var.isAllocated());

  if (var.getNz() == 1) {
    return Field3D(0., this);
  }


  CELL_LOC diffloc = var.getLocation();

  if (this->StaggerGrids && (loc != CELL_DEFAULT) && (loc != var.getLocation())) {
    // Staggered differencing
    throw BoutException("No one used this before. And no one implemented it.");
  }

  Field3D result(this);
  result.allocate(); // Make sure data allocated

  // Check that the input variable has data
  ASSERT1(var.isAllocated());

  stencil s;
  for (const auto &i : result.region(region)) {
    s.c = var[i];
    s.p = var[i.zp()];
    s.m = var[i.zm()];
    s.pp = var[i.offset(0, 0, 2)];
    s.mm = var[i.offset(0, 0, -2)];

    result[i] = func(s);
  }

  result.setLocation(diffloc);

  return result;
}

/*******************************************************************************
 * First central derivatives
 *******************************************************************************/

////////////// X DERIVATIVE /////////////////

const Field3D Mesh::indexDDX(const Field3D &f, CELL_LOC outloc, DIFF_METHOD method, REGION region) {

  Mesh::deriv_func func = fDDX; // Set to default function
  DiffLookup *table = FirstDerivTable;

  CELL_LOC inloc = f.getLocation(); // Input location
  CELL_LOC diffloc = inloc;         // Location of differential result

  Field3D result(this);

  if (this->StaggerGrids && (outloc == CELL_DEFAULT)) {
    // Take care of CELL_DEFAULT case
    outloc = diffloc; // No shift (i.e. same as no stagger case)
  }

  if (this->StaggerGrids && (outloc != inloc)) {
    // Shifting to a new location

    if (((inloc == CELL_CENTRE) && (outloc == CELL_XLOW)) ||
        ((inloc == CELL_XLOW) && (outloc == CELL_CENTRE))) {
      // Shifting in X. Centre -> Xlow, or Xlow -> Centre

      func = sfDDX;                // Set default
      table = FirstStagDerivTable; // Set table for others
      diffloc = (inloc == CELL_CENTRE) ? CELL_XLOW : CELL_CENTRE;

    } else {
      // Derivative of interpolated field or interpolation of derivative field
      // cannot be taken without communicating and applying boundary
      // conditions, so throw an exception instead
      throw BoutException("Unsupported combination of {inloc =%s} and {outloc =%s} in Mesh:indexDDX(Field3D).", strLocation(inloc), strLocation(outloc));
    }
  }

  if (method != DIFF_DEFAULT) {
    // Lookup function
    func = lookupFunc(table, method);
    if (func == nullptr)
      throw BoutException("Cannot use FFT for X derivatives");
  }

  if(this->xstart > 1){
    switch(method) {
    case(DIFF_C2) : {result = applyXdiff<DDX_C2, NG_2_DIR_X>(f, diffloc, region); break;}
    case(DIFF_W2) : {result = applyXdiff<DDX_CWENO2, NG_2_DIR_X>(f, diffloc, region); break;}
    case(DIFF_W3) : {result = applyXdiff<DDX_CWENO3, NG_2_DIR_X>(f, diffloc, region); break;}
    case(DIFF_C4) : {result = applyXdiff<DDX_C4, NG_2_DIR_X>(f, diffloc, region); break;}
    case(DIFF_S2) : {result = applyXdiff<DDX_S2, NG_2_DIR_X>(f, diffloc, region); break;}
    default : throw BoutException("Invalid method encountered in indexDDX");
    }
  } else {
    switch(method) {
    case(DIFF_C2) : {result = applyXdiff<DDX_C2, NG_1_DIR_X>(f, diffloc, region); break;}
    case(DIFF_W2) : {result = applyXdiff<DDX_CWENO2, NG_1_DIR_X>(f, diffloc, region); break;}
    case(DIFF_W3) : {result = applyXdiff<DDX_CWENO3, NG_1_DIR_X>(f, diffloc, region); break;}
    case(DIFF_C4) : {result = applyXdiff<DDX_C4, NG_1_DIR_X>(f, diffloc, region); break;}
    case(DIFF_S2) : {result = applyXdiff<DDX_S2, NG_1_DIR_X>(f, diffloc, region); break;}
    default : throw BoutException("Invalid method encountered in indexDDX");
    }
  }
  
  result.setLocation(diffloc); // Set the result location

  return result;
}

const Field2D Mesh::indexDDX(const Field2D &f, CELL_LOC outloc,
                             DIFF_METHOD method, REGION region) {
  ASSERT1(outloc == CELL_DEFAULT || outloc == f.getLocation());
  ASSERT1(method == DIFF_DEFAULT);
  if(this->xstart > 1){
    switch(method) {
    case(DIFF_C2) : return applyXdiff<DDX_C2, NG_2_DIR_X>(f, outloc, region);
    case(DIFF_W2) : return applyXdiff<DDX_CWENO2, NG_2_DIR_X>(f, outloc, region);
    case(DIFF_W3) : return applyXdiff<DDX_CWENO3, NG_2_DIR_X>(f, outloc, region);
    case(DIFF_C4) : return applyXdiff<DDX_C4, NG_2_DIR_X>(f, outloc, region);
    case(DIFF_S2) : return applyXdiff<DDX_S2, NG_2_DIR_X>(f, outloc, region);
    default : throw BoutException("Invalid method encountered in indexDDX");
    }
  } else {
    switch(method) {
    case(DIFF_C2) : return applyXdiff<DDX_C2, NG_1_DIR_X>(f, outloc, region);
    case(DIFF_W2) : return applyXdiff<DDX_CWENO2, NG_1_DIR_X>(f, outloc, region);
    case(DIFF_W3) : return applyXdiff<DDX_CWENO3, NG_1_DIR_X>(f, outloc, region);
    case(DIFF_C4) : return applyXdiff<DDX_C4, NG_1_DIR_X>(f, outloc, region);
    case(DIFF_S2) : return applyXdiff<DDX_S2, NG_1_DIR_X>(f, outloc, region);
    default : throw BoutException("Invalid method encountered in indexDDX");
    }
  }
}

////////////// Y DERIVATIVE /////////////////

const Field3D Mesh::indexDDY(const Field3D &f, CELL_LOC outloc,
                             DIFF_METHOD method, REGION region) {
  Mesh::deriv_func func = fDDY; // Set to default function
  DiffLookup *table = FirstDerivTable;

  CELL_LOC inloc = f.getLocation(); // Input location
  CELL_LOC diffloc = inloc;         // Location of differential result

  Field3D result(this);

  if (this->StaggerGrids && (outloc == CELL_DEFAULT)) {
    // Take care of CELL_DEFAULT case
    outloc = diffloc; // No shift (i.e. same as no stagger case)
  }

  if (this->StaggerGrids && (outloc != inloc)) {
    // Shifting to a new location
    if (((inloc == CELL_CENTRE) && (outloc == CELL_YLOW)) ||
        ((inloc == CELL_YLOW) && (outloc == CELL_CENTRE))) {
      // Shifting in Y. Centre -> Ylow, or Ylow -> Centre

      func = sfDDY;                // Set default
      table = FirstStagDerivTable; // Set table for others
      diffloc = (inloc == CELL_CENTRE) ? CELL_YLOW : CELL_CENTRE;

    } else {
      // Derivative of interpolated field or interpolation of derivative field
      // cannot be taken without communicating and applying boundary
      // conditions, so throw an exception instead
      throw BoutException("Unsupported combination of {inloc =%s} and {outloc =%s} in Mesh:indexDDY(Field3D).", strLocation(inloc), strLocation(outloc));
    }
  }

  if (method != DIFF_DEFAULT) {
    // Lookup function
    func = lookupFunc(table, method);
    if (func == nullptr)
      throw BoutException("Cannot use FFT for Y derivatives");
  }

  result = applyYdiff(f, func, diffloc, region);

  result.setLocation(diffloc); // Set the result location

  return result;
}

const Field2D Mesh::indexDDY(const Field2D &f, CELL_LOC outloc,
                             DIFF_METHOD method, REGION region) {
  ASSERT1(outloc == CELL_DEFAULT || outloc == f.getLocation());
  ASSERT1(method == DIFF_DEFAULT);
  return applyYdiff(f, fDDY, f.getLocation(), region);
}

////////////// Z DERIVATIVE /////////////////

const Field3D Mesh::indexDDZ(const Field3D &f, CELL_LOC outloc,
                             DIFF_METHOD method, REGION region) {
  Mesh::deriv_func func = fDDZ; // Set to default function
  DiffLookup *table = FirstDerivTable;

  CELL_LOC inloc = f.getLocation(); // Input location
  CELL_LOC diffloc = inloc;         // Location of differential result

  Field3D result(this);

  if (this->StaggerGrids && (outloc == CELL_DEFAULT)) {
    // Take care of CELL_DEFAULT case
    outloc = diffloc; // No shift (i.e. same as no stagger case)
  }

  if (this->StaggerGrids && (outloc != inloc)) {
    // Shifting to a new location

    if (((inloc == CELL_CENTRE) && (outloc == CELL_ZLOW)) ||
        ((inloc == CELL_ZLOW) && (outloc == CELL_CENTRE))) {
      // Shifting in Z. Centre -> Zlow, or Zlow -> Centre

      func = sfDDZ;                // Set default
      table = FirstStagDerivTable; // Set table for others
      diffloc = (inloc == CELL_CENTRE) ? CELL_ZLOW : CELL_CENTRE;

    } else {
      // Derivative of interpolated field or interpolation of derivative field
      // cannot be taken without communicating and applying boundary
      // conditions, so throw an exception instead
      throw BoutException("Unsupported combination of {inloc =%s} and {outloc =%s} in Mesh:indexDDZ(Field3D).", strLocation(inloc), strLocation(outloc));
    }
  }

  if (method != DIFF_DEFAULT) {
    // Lookup function
    func = lookupFunc(table, method);
  }

  if (func == nullptr) {
    // Use FFT

    BoutReal shift = 0.; // Shifting result in Z?
    if (this->StaggerGrids) {
      if ((inloc == CELL_CENTRE) && (diffloc == CELL_ZLOW)) {
        // Shifting down - multiply by exp(-0.5*i*k*dz)
        shift = -1.;
        throw BoutException("Not tested - probably broken");
      } else if ((inloc == CELL_ZLOW) && (diffloc == CELL_CENTRE)) {
        // Shifting up
        shift = 1.;
        throw BoutException("Not tested - probably broken");
      }
    }

    result.allocate(); // Make sure data allocated

    auto region_index = f.region(region);
    int xs = region_index.xstart;
    int xe = region_index.xend;
    int ys = region_index.ystart;
    int ye = region_index.yend;
    ASSERT2(region_index.zstart == 0);
    int ncz = region_index.zend + 1;

    BOUT_OMP(parallel)
    {
      Array<dcomplex> cv(ncz / 2 + 1);


      // Calculate how many Z wavenumbers will be removed
      int kfilter =
          static_cast<int>(fft_derivs_filter * ncz / 2); // truncates, rounding down
      if (kfilter < 0)
        kfilter = 0;
      if (kfilter > (ncz / 2))
        kfilter = ncz / 2;
      int kmax = ncz / 2 - kfilter; // Up to and including this wavenumber index

      BOUT_OMP(for)
      for (int jx = xs; jx <= xe; jx++) {
        for (int jy = ys; jy <= ye; jy++) {
          rfft(f(jx, jy), ncz, cv.begin()); // Forward FFT

          for (int jz = 0; jz <= kmax; jz++) {
            BoutReal kwave = jz * 2.0 * PI / ncz; // wave number is 1/[rad]

            cv[jz] *= dcomplex(0.0, kwave);
            if (shift)
              cv[jz] *= exp(Im * (shift * kwave));
          }
          for (int jz = kmax + 1; jz <= ncz / 2; jz++) {
            cv[jz] = 0.0;
          }

          irfft(cv.begin(), ncz, result(jx, jy)); // Reverse FFT
        }
      }
    }
      // End of parallel section

#if CHECK > 0
    // Mark boundaries as invalid
    result.bndry_xin = false;
    result.bndry_xout = false;
    result.bndry_yup = false;
    result.bndry_ydown = false;
#endif

  } else {
    // All other (non-FFT) functions
    result = applyZdiff(f, func, diffloc, region);
  }

  result.setLocation(diffloc);

  return result;
}

const Field2D Mesh::indexDDZ(const Field2D &f, CELL_LOC UNUSED(outloc),
                             DIFF_METHOD UNUSED(method), REGION UNUSED(region)) {
  ASSERT1(this == f.getMesh());
  return Field2D(0., this);
}

/*******************************************************************************
 * 2nd derivatives
 *******************************************************************************/

////////////// X DERIVATIVE /////////////////

/*!
 * @brief Calculates second X derivative on Mesh in index space
 *
 * @param[in] f        3D scalar field to be differentiated.
 *                     Must be allocated and finite
 *
 * @param[in] outloc   The cell location of the result
 *
 * @param[in] method   The numerical method to use
 *
 * @return  A 3D scalar field with invalid data in the
 *          guard cells
 *
 */
const Field3D Mesh::indexD2DX2(const Field3D &f, CELL_LOC outloc,
                               DIFF_METHOD method, REGION region) {
  Mesh::deriv_func func = fD2DX2; // Set to default function
  DiffLookup *table = SecondDerivTable;

  CELL_LOC inloc = f.getLocation(); // Input location
  CELL_LOC diffloc = inloc;         // Location of differential result

  ASSERT1(this == f.getMesh());

  Field3D result(this);

  if (StaggerGrids && (outloc == CELL_DEFAULT)) {
    // Take care of CELL_DEFAULT case
    outloc = diffloc; // No shift (i.e. same as no stagger case)
  }

  if (StaggerGrids && (outloc != inloc)) {
    // Shifting to a new location

    if (((inloc == CELL_CENTRE) && (outloc == CELL_XLOW)) ||
        ((inloc == CELL_XLOW) && (outloc == CELL_CENTRE))) {
      // Shifting in X. Centre -> Xlow, or Xlow -> Centre

      func = sfD2DX2;               // Set default
      table = SecondStagDerivTable; // Set table for others
      diffloc = (inloc == CELL_CENTRE) ? CELL_XLOW : CELL_CENTRE;

    } else {
      // Derivative of interpolated field or interpolation of derivative field
      // cannot be taken without communicating and applying boundary
      // conditions, so throw an exception instead
      throw BoutException("Unsupported combination of {inloc =%s} and {outloc =%s} in Mesh:indexD2DX2(Field3D).", strLocation(inloc), strLocation(outloc));
    }
  }

  if (method != DIFF_DEFAULT) {
    // Lookup function
    func = lookupFunc(table, method);
    if (func == nullptr)
      throw BoutException("Cannot use FFT for X derivatives");
  }

  result = applyXdiff(f, func, diffloc, region);

  result.setLocation(diffloc);

  return result;
}

/*!
 * @brief Calculates second X derivative on Mesh in index space
 *
 * @param[in] f        2D scalar field to be differentiated.
 *                     Must be allocated and finite
 *
 * @return  A 2D scalar field with invalid data in the
 *          guard cells
 *
 */
const Field2D Mesh::indexD2DX2(const Field2D &f,  CELL_LOC outloc,
                               DIFF_METHOD method, REGION region) {
  ASSERT1(outloc == CELL_DEFAULT || outloc == f.getLocation());
  ASSERT1(method == DIFF_DEFAULT);
  return applyXdiff(f, fD2DX2, f.getLocation(), region);
}

////////////// Y DERIVATIVE /////////////////

/*!
 * @brief Calculates second Y derivative on Mesh in index space
 *
 * @param[in] f        3D scalar field to be differentiated.
 *                     Must be allocated and finite
 *
 * @return  A 3D scalar field with invalid data in the
 *          guard cells
 *
 */
const Field3D Mesh::indexD2DY2(const Field3D &f, CELL_LOC outloc,
                               DIFF_METHOD method, REGION region) {
  Mesh::deriv_func func = fD2DY2; // Set to default function
  DiffLookup *table = SecondDerivTable;

  CELL_LOC inloc = f.getLocation(); // Input location
  CELL_LOC diffloc = inloc;         // Location of differential result

  ASSERT1(this == f.getMesh());

  Field3D result(this);

  if (StaggerGrids && (outloc == CELL_DEFAULT)) {
    // Take care of CELL_DEFAULT case
    outloc = diffloc; // No shift (i.e. same as no stagger case)
  }

  if (StaggerGrids && (outloc != inloc)) {
    // Shifting to a new location

    if (((inloc == CELL_CENTRE) && (outloc == CELL_YLOW)) ||
        ((inloc == CELL_YLOW) && (outloc == CELL_CENTRE))) {
      // Shifting in Y. Centre -> Ylow, or Ylow -> Centre

      func = sfD2DY2;               // Set default
      table = SecondStagDerivTable; // Set table for others
      diffloc = (inloc == CELL_CENTRE) ? CELL_YLOW : CELL_CENTRE;

    } else {
      // Derivative of interpolated field or interpolation of derivative field
      // cannot be taken without communicating and applying boundary
      // conditions, so throw an exception instead
      throw BoutException("Unsupported combination of {inloc =%s} and {outloc =%s} in Mesh:indexD2DY2(Field3D).", strLocation(inloc), strLocation(outloc));
    }
  }

  if (method != DIFF_DEFAULT) {
    // Lookup function
    func = lookupFunc(table, method);
    if (func == nullptr)
      throw BoutException("Cannot use FFT for Y derivatives");
  }

  result = applyYdiff(f, func, diffloc, region);

  result.setLocation(diffloc);

  return result;
}

/*!
 * @brief Calculates second Y derivative on Mesh in index space
 *
 * @param[in] f        2D scalar field to be differentiated.
 *                     Must be allocated and finite
 *
 * @return  A 2D scalar field with invalid data in the
 *          guard cells
 *
 */
const Field2D Mesh::indexD2DY2(const Field2D &f, CELL_LOC outloc,
                               DIFF_METHOD method, REGION region) {
  ASSERT1(outloc == CELL_DEFAULT || outloc == f.getLocation());
  ASSERT1(method == DIFF_DEFAULT);
  return applyYdiff(f, fD2DY2, f.getLocation(), region);
}

////////////// Z DERIVATIVE /////////////////

/*!
 * @brief Calculates second Z derivative on Mesh in index space
 *
 * @param[in] f        3D scalar field to be differentiated.
 *                     Must be allocated and finite
 *
 * @return  A 3D scalar field with invalid data in the
 *          guard cells
 *
 */
const Field3D Mesh::indexD2DZ2(const Field3D &f, CELL_LOC outloc,
                               DIFF_METHOD method, REGION region) {
  Mesh::deriv_func func = fD2DZ2; // Set to default function
  DiffLookup *table = SecondDerivTable;

  CELL_LOC inloc = f.getLocation(); // Input location
  CELL_LOC diffloc = inloc;         // Location of differential result

  ASSERT1(this == f.getMesh());

  Field3D result(this);

  if (StaggerGrids && (outloc == CELL_DEFAULT)) {
    // Take care of CELL_DEFAULT case
    outloc = diffloc; // No shift (i.e. same as no stagger case)
  }

  if (StaggerGrids && (outloc != inloc)) {
    // Shifting to a new location

    if (((inloc == CELL_CENTRE) && (outloc == CELL_ZLOW)) ||
        ((inloc == CELL_ZLOW) && (outloc == CELL_CENTRE))) {
      // Shifting in Z. Centre -> Zlow, or Zlow -> Centre

      func = sfD2DZ2;               // Set default
      table = SecondStagDerivTable; // Set table for others
      diffloc = (inloc == CELL_CENTRE) ? CELL_ZLOW : CELL_CENTRE;

    } else {
      // Derivative of interpolated field or interpolation of derivative field
      // cannot be taken without communicating and applying boundary
      // conditions, so throw an exception instead
      throw BoutException("Unsupported combination of {inloc =%s} and {outloc =%s} in Mesh:indexD2DZ2(Field3D).", strLocation(inloc), strLocation(outloc));
    }
  }

  if (method != DIFF_DEFAULT) {
    // Lookup function
    func = lookupFunc(table, method);
  }

  if (func == nullptr) {
    // Use FFT

    BoutReal shift = 0.; // Shifting result in Z?
    if (StaggerGrids) {
      if ((inloc == CELL_CENTRE) && (diffloc == CELL_ZLOW)) {
	      // Shifting down - multiply by exp(-0.5*i*k*dz) 
        throw BoutException("Not tested - probably broken");
      } else if((inloc == CELL_ZLOW) && (diffloc == CELL_CENTRE)) {
	      // Shifting up
        throw BoutException("Not tested - probably broken");

      } else if (diffloc != CELL_DEFAULT && diffloc != inloc){
        throw BoutException("Not implemented!");
      }
    }

    result.allocate(); // Make sure data allocated

    auto region_index = f.region(region);
    int xs = region_index.xstart;
    int xe = region_index.xend;
    int ys = region_index.ystart;
    int ye = region_index.yend;
    ASSERT2(region_index.zstart == 0);
    int ncz = region_index.zend + 1;

    // TODO: The comment does not match the check
    ASSERT1(ncz % 2 == 0); // Must be a power of 2
    Array<dcomplex> cv(ncz / 2 + 1);
    
    for (int jx = xs; jx <= xe; jx++) {
      for (int jy = ys; jy <= ye; jy++) {

        rfft(f(jx, jy), ncz, cv.begin()); // Forward FFT

        for (int jz = 0; jz <= ncz / 2; jz++) {
          BoutReal kwave = jz * 2.0 * PI / ncz; // wave number is 1/[rad]

          cv[jz] *= -SQ(kwave);
          if (shift)
            cv[jz] *= exp(0.5 * Im * (shift * kwave));
        }

        irfft(cv.begin(), ncz, result(jx, jy)); // Reverse FFT
      }
    }

#if CHECK > 0
    // Mark boundaries as invalid
    result.bndry_xin = false;
    result.bndry_xout = false;
    result.bndry_yup = false;
    result.bndry_ydown = false;
#endif

  } else {
    // All other (non-FFT) functions
    result = applyZdiff(f, func, diffloc, region);
  }

  result.setLocation(diffloc);

  return result;
}

/*******************************************************************************
 * Fourth derivatives
 *******************************************************************************/

const Field3D Mesh::indexD4DX4(const Field3D &f, CELL_LOC outloc,
                               DIFF_METHOD method, REGION region) {
  ASSERT1(outloc == CELL_DEFAULT || outloc == f.getLocation());
  ASSERT1(method == DIFF_DEFAULT);
  return applyXdiff(f, D4DX4_C2, f.getLocation(), region);
}

const Field2D Mesh::indexD4DX4(const Field2D &f, CELL_LOC outloc,
                               DIFF_METHOD method, REGION region) {
  ASSERT1(outloc == CELL_DEFAULT || outloc == f.getLocation());
  ASSERT1(method == DIFF_DEFAULT);
  return applyXdiff(f, D4DX4_C2, f.getLocation(), region);
}

const Field3D Mesh::indexD4DY4(const Field3D &f, CELL_LOC outloc,
                               DIFF_METHOD method, REGION region) {
  ASSERT1(outloc == CELL_DEFAULT || outloc == f.getLocation());
  ASSERT1(method == DIFF_DEFAULT);
  return applyYdiff(f, D4DX4_C2, f.getLocation(), region);
}

const Field2D Mesh::indexD4DY4(const Field2D &f, CELL_LOC outloc,
                               DIFF_METHOD method, REGION region) {
  ASSERT1(outloc == CELL_DEFAULT || outloc == f.getLocation());
  ASSERT1(method == DIFF_DEFAULT);
  return applyYdiff(f, D4DX4_C2, f.getLocation(), region);
}

const Field3D Mesh::indexD4DZ4(const Field3D &f, CELL_LOC outloc,
                               DIFF_METHOD method, REGION region){
  ASSERT1(outloc == CELL_DEFAULT || outloc == f.getLocation());
  ASSERT1(method == DIFF_DEFAULT);
  return applyZdiff(f, D4DX4_C2, f.getLocation(), region);
}

const Field2D Mesh::indexD4DZ4(const Field2D &f, CELL_LOC outloc,
                               DIFF_METHOD UNUSED(method), REGION UNUSED(region)) {
  ASSERT1(outloc == CELL_DEFAULT || outloc == f.getLocation());
  return Field2D(0., this);
}

/*******************************************************************************
 * Mixed derivatives
 *******************************************************************************/

/*******************************************************************************
 * Advection schemes
 *
 * Jan 2018  - Re-written to use iterators and handle staggering as different cases
 * Jan 2009  - Re-written to use Set*Stencil routines
 *******************************************************************************/

////////////// X DERIVATIVE /////////////////

/// Special case where both arguments are 2D. Output location ignored for now
const Field2D Mesh::indexVDDX(const Field2D &v, const Field2D &f, CELL_LOC outloc,
                              DIFF_METHOD method, REGION region) {
  TRACE("Mesh::indexVDDX(Field2D, Field2D)");

  CELL_LOC diffloc = f.getLocation();

  Mesh::upwind_func func = fVDDX;

  if (method != DIFF_DEFAULT) {
    // Lookup function
    func = lookupFunc(UpwindTable, method);
  }

  ASSERT1(this == f.getMesh());
  ASSERT1(this == v.getMesh());
  ASSERT2((v.getLocation() == f.getLocation()) && ((outloc == CELL_DEFAULT) || (outloc == f.getLocation()))); // No staggering allowed for Field2D

  Field2D result(this);
  result.allocate(); // Make sure data allocated

  if (this->xstart > 1) {
    // Two or more guard cells

    stencil s;
    for (const auto &i : result.region(region)) {
      s.c = f[i];
      s.p = f[i.xp()];
      s.m = f[i.xm()];
      s.pp = f[i.offset(2, 0, 0)];
      s.mm = f[i.offset(-2, 0, 0)];

      result[i] = func(v[i], s);
    }

  } else if (this->xstart == 1) {
    // Only one guard cell

    stencil s;
    s.pp = nan("");
    s.mm = nan("");

    for (const auto &i : result.region(region)) {
      s.c = f[i];
      s.p = f[i.xp()];
      s.m = f[i.xm()];

      result[i] = func(v[i], s);
    }
  } else {
    // No guard cells
    throw BoutException("Error: Derivatives in X requires at least one guard cell");
  }

#if CHECK > 0
  // Mark boundaries as invalid
  result.bndry_xin = result.bndry_xout = false;
#endif

  result.setLocation(diffloc);

  return result;
}

/// General version for 3D objects.
/// 2D objects passed as input will result in copying
const Field3D Mesh::indexVDDX(const Field3D &v, const Field3D &f, CELL_LOC outloc,
                              DIFF_METHOD method, REGION region) {
  TRACE("Mesh::indexVDDX(Field3D, Field3D)");

  ASSERT1(this == v.getMesh());
  ASSERT1(this == f.getMesh());

  Field3D result(this);
  result.allocate(); // Make sure data allocated

  CELL_LOC vloc = v.getLocation();
  CELL_LOC inloc = f.getLocation(); // Input location
  CELL_LOC diffloc = inloc;         // Location of differential result

  if (StaggerGrids && (outloc == CELL_DEFAULT)) {
    // Take care of CELL_DEFAULT case
    outloc = diffloc; // No shift (i.e. same as no stagger case)
  }

  if (StaggerGrids && (vloc != inloc)) {
    // Staggered grids enabled, and velocity at different location to value

    Mesh::flux_func func = sfVDDX;
    DiffLookup *table = UpwindTable;

    if (vloc == CELL_XLOW) {
      // V staggered w.r.t. variable
      func = sfVDDX;
      table = UpwindStagTable;
      diffloc = CELL_CENTRE;
    } else if ((vloc == CELL_CENTRE) && (inloc == CELL_XLOW)) {
      // Shifted
      func = sfVDDX;
      table = UpwindStagTable;
      diffloc = CELL_XLOW;
    } else {
      // More complicated shifting. The user should probably
      // be explicit about what interpolation should be done

      throw BoutException("Unhandled shift in Mesh::indexVDDX");
    }

    if (method != DIFF_DEFAULT) {
      // Lookup function
      func = lookupFunc(table, method);
    }

    // Note: The velocity stencil contains only (mm, m, p, pp)
    // v.p is v at +1/2, v.m is at -1/2 relative to the field f

    if (this->xstart > 1) {
      // Two or more guard cells

      if ((vloc == CELL_XLOW) && (diffloc == CELL_CENTRE)) {
        stencil fs, vs;
        vs.c = nan("");

        for (const auto &i : result.region(region)) {
          fs.c = f[i];
          fs.p = f[i.xp()];
          fs.m = f[i.xm()];
          fs.pp = f[i.offset(2, 0, 0)];
          fs.mm = f[i.offset(-2, 0, 0)];

          vs.mm = v[i.xm()];
          vs.m = v[i];
          vs.p = v[i.xp()];
          vs.pp = v[i.offset(2, 0, 0)];

          result[i] = func(vs, fs);
        }

      } else if ((vloc == CELL_CENTRE) && (diffloc == CELL_XLOW)) {
        stencil fs, vs;
        vs.c = nan("");

        for (const auto &i : result.region(region)) {
          fs.c = f[i];
          fs.p = f[i.xp()];
          fs.m = f[i.xm()];
          fs.pp = f[i.offset(2, 0, 0)];
          fs.mm = f[i.offset(-2, 0, 0)];

          vs.mm = v[i.offset(-2, 0, 0)];
          vs.m = v[i.xm()];
          vs.p = v[i];
          vs.pp = v[i.xp()];

          result[i] = func(vs, fs);
        }
      } else {
        throw BoutException("Unhandled shift in Mesh::indexVDDX");
      }
    } else if (this->xstart == 1) {
      // One guard cell

      if ((vloc == CELL_XLOW) && (diffloc == CELL_CENTRE)) {
        stencil fs, vs;
        vs.c = nan("");
        vs.pp = nan("");
        fs.pp = nan("");
        fs.mm = nan("");

        for (const auto &i : result.region(region)) {
          fs.c = f[i];
          fs.p = f[i.xp()];
          fs.m = f[i.xm()];

          vs.mm = v[i.xm()];
          vs.m = v[i];
          vs.p = v[i.xp()];

          result[i] = func(vs, fs);
        }

      } else if ((vloc == CELL_CENTRE) && (diffloc == CELL_XLOW)) {
        stencil fs, vs;

        fs.pp = nan("");
        fs.mm = nan("");
        vs.c = nan("");
        vs.mm = nan("");

        for (const auto &i : result.region(region)) {
          fs.c = f[i];
          fs.p = f[i.xp()];
          fs.m = f[i.xm()];

          vs.m = v[i.xm()];
          vs.p = v[i];
          vs.pp = v[i.xp()];

          result[i] = func(vs, fs);
        }
      } else {
        throw BoutException("Unhandled shift in Mesh::indexVDDX");
      }
    } else {
      // No guard cells
      throw BoutException("Error: Derivatives in X requires at least one guard cell");
    }

  } else {
    // Not staggered
    Mesh::upwind_func func = fVDDX;
    DiffLookup *table = UpwindTable;

    if (method != DIFF_DEFAULT) {
      // Lookup function
      func = lookupFunc(table, method);
    }

    if (this->xstart > 1) {
      // Two or more guard cells
      stencil fs;
      for (const auto &i : result.region(region)) {
        fs.c = f[i];
        fs.p = f[i.xp()];
        fs.m = f[i.xm()];
        fs.pp = f[i.offset(2, 0, 0)];
        fs.mm = f[i.offset(-2, 0, 0)];

        result[i] = func(v[i], fs);
      }
    } else if (this->xstart == 1) {
      // Only one guard cell
      stencil fs;
      fs.pp = nan("");
      fs.mm = nan("");
      for (const auto &i : result.region(region)) {
        fs.c = f[i];
        fs.p = f[i.xp()];
        fs.m = f[i.xm()];

        result[i] = func(v[i], fs);
      }
    } else {
      // No guard cells
      throw BoutException("Error: Derivatives in X requires at least one guard cell");
    }
  }

  result.setLocation(diffloc);

#if CHECK > 0
  // Mark boundaries as invalid
  result.bndry_xin = result.bndry_xout = result.bndry_yup = result.bndry_ydown = false;
#endif

  return result;
}

////////////// Y DERIVATIVE /////////////////

// special case where both are 2D
const Field2D Mesh::indexVDDY(const Field2D &v, const Field2D &f, CELL_LOC outloc,
                              DIFF_METHOD method, REGION region) {
  TRACE("Mesh::indexVDDY");

  ASSERT1(this == v.getMesh());
  ASSERT1(this == f.getMesh());
  ASSERT2((v.getLocation() == f.getLocation()) && ((outloc == CELL_DEFAULT) || (outloc == f.getLocation()))); // No staggering allowed for Field2D

  Field2D result(this);
  result.allocate(); // Make sure data allocated

  CELL_LOC vloc = v.getLocation();
  CELL_LOC inloc = f.getLocation(); // Input location
  CELL_LOC diffloc = inloc;         // Location of differential result

  if (outloc == CELL_DEFAULT) {
    // Take care of CELL_DEFAULT case
    outloc = diffloc; // No shift (i.e. same as no stagger case)
  }

  if (this->LocalNy == 1){
    result=0;
    result.setLocation(outloc);
    return result;
  }

  ASSERT1(this->ystart > 0); // Must have at least one guard cell

  if (StaggerGrids && (vloc != inloc)) {
    // Staggered grids enabled, and velocity at different location to value

    Mesh::flux_func func = sfVDDY;
    DiffLookup *table = UpwindTable;
    if ((vloc == CELL_YLOW) && (diffloc == CELL_CENTRE)) {
      // V staggered w.r.t. variable
      func = sfVDDY;
      table = UpwindStagTable;
    } else if ((vloc == CELL_CENTRE) && (inloc == CELL_YLOW)) {
      // Shifted
      func = sfVDDY;
      table = UpwindStagTable;
      diffloc = CELL_YLOW;
    } else {
      // More complicated. Deciding what to do here isn't straightforward

      throw BoutException("Unhandled shift in indexVDDY(Field2D, Field2D)");
    }

    if (method != DIFF_DEFAULT) {
      // Lookup function
      func = lookupFunc(table, method);
    }

    // Note: vs.c not used for staggered differencing
    // vs.m is at i-1/2, vs.p is as i+1/2
    if ((vloc == CELL_YLOW) && (diffloc == CELL_CENTRE)) {
      if (this->ystart > 1) {
        // Two or more guard cells
        stencil fs, vs;
        vs.c = nan("");
        for (const auto &i : result.region(region)) {

          fs.c = f[i];
          fs.p = f[i.yp()];
          fs.m = f[i.ym()];
          fs.pp = f[i.offset(0, 2, 0)];
          fs.mm = f[i.offset(0, -2, 0)];

          vs.pp = v[i.offset(0, 2, 0)];
          vs.p = v[i.yp()];
          vs.m = v[i];
          vs.mm = v[i.ym()];

          result[i] = func(vs, fs);
        }

      } else {
        // Only one guard cell

        stencil fs, vs;
        fs.pp = nan("");
        fs.mm = nan("");
        vs.c = nan("");
        vs.pp = nan("");

        for (const auto &i : result.region(region)) {
          fs.c = f[i];
          fs.p = f[i.yp()];
          fs.m = f[i.ym()];

          vs.p = v[i.yp()];
          vs.m = v[i];
          vs.mm = v[i.ym()];

          result[i] = func(vs, fs);
        }
      }
    } else if ((vloc == CELL_CENTRE) && (inloc == CELL_YLOW)) {
      if (this->ystart > 1) {
        // Two or more guard cells
        stencil fs, vs;
        vs.c = nan("");
        for (const auto &i : result.region(region)) {

          fs.c = f[i];
          fs.p = f[i.yp()];
          fs.m = f[i.ym()];
          fs.pp = f[i.offset(0, 2, 0)];
          fs.mm = f[i.offset(0, -2, 0)];

          vs.pp = v[i.yp()];
          vs.p = v[i];
          vs.m = v[i.ym()];
          vs.mm = v[i.offset(0, -2, 0)];

          result[i] = func(vs, fs);
        }

      } else {
        // Only one guard cell

        stencil fs, vs;
        fs.pp = nan("");
        fs.mm = nan("");
        vs.c = nan("");
        vs.mm = nan("");

        for (const auto &i : result.region(region)) {
          fs.c = f[i];
          fs.p = f[i.yp()];
          fs.m = f[i.ym()];

          vs.pp = v[i.yp()];
          vs.p = v[i];
          vs.m = v[i.ym()];

          result[i] = func(vs, fs);
        }
      }
    } else {
      throw BoutException("Unhandled shift in indexVDDY(Field2D, Field2D)");
    }

  } else {
    // Not staggered

    Mesh::upwind_func func = fVDDY;
    DiffLookup *table = UpwindTable;

    if (method != DIFF_DEFAULT) {
      // Lookup function
      func = lookupFunc(table, method);
    }

    if (this->ystart > 1) {
      // Two or more guard cells
      stencil fs;
      for (const auto &i : result.region(region)) {

        fs.c = f[i];
        fs.p = f[i.yp()];
        fs.m = f[i.ym()];
        fs.pp = f[i.offset(0, 2, 0)];
        fs.mm = f[i.offset(0, -2, 0)];

        result[i] = func(v[i], fs);
      }

    } else {
      // Only one guard cell

      stencil fs;
      fs.pp = nan("");
      fs.mm = nan("");

      for (const auto &i : result.region(region)) {
        fs.c = f[i];
        fs.p = f[i.yp()];
        fs.m = f[i.ym()];

        result[i] = func(v[i], fs);
      }
    }
  }

  result.setLocation(diffloc);

#if CHECK > 0
  // Mark boundaries as invalid
  result.bndry_xin = result.bndry_xout = result.bndry_yup = result.bndry_ydown = false;
#endif

  return result;
}

// general case
const Field3D Mesh::indexVDDY(const Field3D &v, const Field3D &f, CELL_LOC outloc,
                              DIFF_METHOD method, REGION region) {
  TRACE("Mesh::indexVDDY(Field3D, Field3D)");

  ASSERT1(this == v.getMesh());
  ASSERT1(this == f.getMesh());

  Field3D result(this);
  result.allocate(); // Make sure data allocated

  CELL_LOC vloc = v.getLocation();
  CELL_LOC inloc = f.getLocation(); // Input location
  CELL_LOC diffloc = inloc;         // Location of differential result

  if (outloc == CELL_DEFAULT) {
    // Take care of CELL_DEFAULT case
    outloc = diffloc; // No shift (i.e. same as no stagger case)
  }

  if (this->LocalNy == 1){
    result=0;
    result.setLocation(outloc);
    return result;
  }

  ASSERT1(this->ystart > 0); // Need at least one guard cell

  if (StaggerGrids && (vloc != inloc)) {
    // Staggered grids enabled, and velocity at different location to value

    Mesh::flux_func func = sfVDDY;
    DiffLookup *table = UpwindTable;

    if (vloc == CELL_YLOW) {
      // V staggered w.r.t. variable
      func = sfVDDY;
      table = UpwindStagTable;
      diffloc = CELL_CENTRE;
    } else if ((vloc == CELL_CENTRE) && (inloc == CELL_YLOW)) {
      // Shifted
      func = sfVDDY;
      table = UpwindStagTable;
      diffloc = CELL_YLOW;
    } else {
      // More complicated. Deciding what to do here isn't straightforward

      throw BoutException("Unhandled shift in VDDY(Field, Field)");
    }

    if (method != DIFF_DEFAULT) {
      // Lookup function
      func = lookupFunc(table, method);
    }

    // There are four cases, corresponding to whether or not f and v
    // have yup, ydown fields.

    // If vUseUpDown is true, field "v" has distinct yup and ydown fields which
    // will be used to calculate a derivative along
    // the magnetic field
    bool vUseUpDown = (v.hasYupYdown() && ((&v.yup() != &v) || (&v.ydown() != &v)));
    bool fUseUpDown = (f.hasYupYdown() && ((&f.yup() != &f) || (&f.ydown() != &f)));

    if (vUseUpDown && fUseUpDown) {
      // Both v and f have up/down fields

      stencil vval, fval;
      vval.pp = nan("");
      vval.mm = nan("");
      fval.pp = nan("");
      fval.mm = nan("");
      for (const auto &i : result.region(region)) {
        vval.c = v[i];
        vval.p = v.yup()[i.yp()];
        vval.m = v.ydown()[i.ym()];
        fval.c = f[i];
        fval.p = f.yup()[i.yp()];
        fval.m = f.ydown()[i.ym()];

        if (diffloc != CELL_DEFAULT) {
          // Non-centred stencil
          if ((vloc == CELL_CENTRE) && (diffloc == CELL_YLOW)) {
            // Producing a stencil centred around a lower Y value
            vval.pp = vval.p;
            vval.p = vval.c;
          } else if (vloc == CELL_YLOW) {
            // Stencil centred around a cell centre
            vval.mm = vval.m;
            vval.m = vval.c;
          }
          // Shifted in one direction -> shift in another
          // Could produce warning
        }
        result[i] = func(vval, fval);
      }
    } else if (vUseUpDown) {
      // Only v has up/down fields
      // f must shift to field aligned coordinates
      Field3D f_fa = this->toFieldAligned(f);

      stencil vval, fval;
      vval.pp = nan("");
      vval.mm = nan("");
      for (const auto &i : result.region(region)) {
        vval.c = v[i];
        vval.p = v.yup()[i.yp()];
        vval.m = v.ydown()[i.ym()];
        fval.c = f_fa[i];
        fval.p = f_fa[i.yp()];
        fval.m = f_fa[i.ym()];
        fval.pp = f_fa[i.offset(0, 2, 0)];
        fval.mm = f_fa[i.offset(0, -2, 0)];

        if (diffloc != CELL_DEFAULT) {
          // Non-centred stencil
          if ((vloc == CELL_CENTRE) && (diffloc == CELL_YLOW)) {
            // Producing a stencil centred around a lower Y value
            vval.pp = vval.p;
            vval.p = vval.c;
          } else if (vloc == CELL_YLOW) {
            // Stencil centred around a cell centre
            vval.mm = vval.m;
            vval.m = vval.c;
          }
          // Shifted in one direction -> shift in another
          // Could produce warning
        }
        result[i] = func(vval, fval);
      }
    } else if (fUseUpDown) {
      // Only f has up/down fields
      // v must shift to field aligned coordinates
      Field3D v_fa = this->toFieldAligned(v);

      stencil vval, fval;
      fval.pp = nan("");
      fval.mm = nan("");
      for (const auto &i : result.region(region)) {
        vval.c = v_fa[i];
        vval.p = v_fa[i.yp()];
        vval.m = v_fa[i.ym()];
        vval.pp = v_fa[i.offset(0, 2, 0)];
        vval.mm = v_fa[i.offset(0, -2, 0)];
        fval.c = f[i];
        fval.p = f.yup()[i.yp()];
        fval.m = f.ydown()[i.ym()];

        if (diffloc != CELL_DEFAULT) {
          // Non-centred stencil
          if ((vloc == CELL_CENTRE) && (diffloc == CELL_YLOW)) {
            // Producing a stencil centred around a lower Y value
            vval.pp = vval.p;
            vval.p = vval.c;
          } else if (vloc == CELL_YLOW) {
            // Stencil centred around a cell centre
            vval.mm = vval.m;
            vval.m = vval.c;
          }
          // Shifted in one direction -> shift in another
          // Could produce warning
        }
        result[i] = func(vval, fval);
      }
    } else {
      // Both must shift to field aligned
      Field3D v_fa = this->toFieldAligned(v);
      Field3D f_fa = this->toFieldAligned(f);

      stencil vval, fval;
      for (const auto &i : result.region(region)) {
        vval.c = v_fa[i];
        vval.p = v_fa[i.yp()];
        vval.m = v_fa[i.ym()];
        vval.pp = v_fa[i.offset(0, 2, 0)];
        vval.mm = v_fa[i.offset(0, -2, 0)];
        fval.c = f[i];
        fval.p = f_fa[i.yp()];
        fval.m = f_fa[i.ym()];
        fval.pp = f_fa[i.offset(0, 2, 0)];
        fval.mm = f_fa[i.offset(0, -2, 0)];

        if (diffloc != CELL_DEFAULT) {
          // Non-centred stencil
          if ((vloc == CELL_CENTRE) && (diffloc == CELL_YLOW)) {
            // Producing a stencil centred around a lower Y value
            vval.pp = vval.p;
            vval.p = vval.c;
          } else if (vloc == CELL_YLOW) {
            // Stencil centred around a cell centre
            vval.mm = vval.m;
            vval.m = vval.c;
          }
          // Shifted in one direction -> shift in another
          // Could produce warning
        }
        result[i] = func(vval, fval);
      }
    }
  } else {
    // Non-staggered case

    Mesh::upwind_func func = fVDDY;
    DiffLookup *table = UpwindTable;

    if (method != DIFF_DEFAULT) {
      // Lookup function
      func = lookupFunc(table, method);
    }

    if (f.hasYupYdown() && ((&f.yup() != &f) || (&f.ydown() != &f))) {
      // f has yup and ydown fields which are distinct

      stencil fs;
      fs.pp = nan("");
      fs.mm = nan("");

      Field3D f_yup = f.yup();
      Field3D f_ydown = f.ydown();

      for (const auto &i : result.region(region)) {

        fs.c = f[i];
        fs.p = f_yup[i.yp()];
        fs.m = f_ydown[i.ym()];

        result[i] = func(v[i], fs);
      }

    } else {
      // Not using yup/ydown fields, so first transform to field-aligned coordinates

      Field3D f_fa = this->toFieldAligned(f);
      Field3D v_fa = this->toFieldAligned(v);

      if (this->ystart > 1) {
        stencil fs;

        for (const auto &i : result.region(region)) {
          fs.c = f_fa[i];
          fs.p = f_fa[i.yp()];
          fs.m = f_fa[i.ym()];
          fs.pp = f_fa[i.offset(0, 2, 0)];
          fs.mm = f_fa[i.offset(0, -2, 0)];

          result[i] = func(v_fa[i], fs);
        }
      } else {
        stencil fs;
        fs.pp = nan("");
        fs.mm = nan("");

        for (const auto &i : result.region(region)) {
          fs.c = f_fa[i];
          fs.p = f_fa[i.yp()];
          fs.m = f_fa[i.ym()];

          result[i] = func(v_fa[i], fs);
        }
      }
      // Shift result back
      result = this->fromFieldAligned(result);
    }
  }

  result.setLocation(diffloc);

#if CHECK > 0
  // Mark boundaries as invalid
  result.bndry_xin = result.bndry_xout = result.bndry_yup = result.bndry_ydown = false;
#endif

  return result;
}

////////////// Z DERIVATIVE /////////////////

// general case
const Field3D Mesh::indexVDDZ(const Field3D &v, const Field3D &f, CELL_LOC outloc,
                              DIFF_METHOD method, REGION region) {
  TRACE("Mesh::indexVDDZ");

  ASSERT1(this == v.getMesh());
  ASSERT1(this == f.getMesh());

  Field3D result(this);
  result.allocate(); // Make sure data allocated

  CELL_LOC vloc = v.getLocation();
  CELL_LOC inloc = f.getLocation(); // Input location
  CELL_LOC diffloc = inloc;         // Location of differential result

  if (StaggerGrids && (outloc == CELL_DEFAULT)) {
    // Take care of CELL_DEFAULT case
    outloc = diffloc; // No shift (i.e. same as no stagger case)
  }

  if (StaggerGrids && (vloc != inloc)) {
    // Staggered grids enabled, and velocity at different location to value

    Mesh::flux_func func = sfVDDZ;
    DiffLookup *table = UpwindTable;

    if (vloc == CELL_ZLOW) {
      // V staggered w.r.t. variable
      func = sfVDDZ;
      table = UpwindStagTable;
      diffloc = CELL_CENTRE;
    } else if ((vloc == CELL_CENTRE) && (inloc == CELL_ZLOW)) {
      // Shifted
      func = sfVDDZ;
      table = UpwindStagTable;
      diffloc = CELL_ZLOW;
    } else {
      // More complicated. Deciding what to do here isn't straightforward

      throw BoutException("Unhandled shift in indexVDDZ");
    }

    if (method != DIFF_DEFAULT) {
      // Lookup function
      func = lookupFunc(table, method);
    }

    stencil vval, fval;
    for (const auto &i : result.region(region)) {
      fval.mm = f[i.offset(0,0,-2)];
      fval.m = f[i.zm()];
      fval.c = f[i];
      fval.p = f[i.zp()];
      fval.pp = f[i.offset(0,0,2)];

      vval.mm = v[i.offset(0,0,-2)];
      vval.m = v[i.zm()];
      vval.c = v[i];
      vval.p = v[i.zp()];
      vval.pp = v[i.offset(0,0,2)];

      if((diffloc != CELL_DEFAULT) && (diffloc != vloc)) {
        // Non-centred stencil

        if((vloc == CELL_CENTRE) && (diffloc == CELL_ZLOW)) {
          // Producing a stencil centred around a lower Z value
          vval.pp = vval.p;
          vval.p  = vval.c;

        }else if(vloc == CELL_ZLOW) {
          // Stencil centred around a cell centre

          vval.mm = vval.m;
          vval.m  = vval.c;
        }
        // Shifted in one direction -> shift in another
        // Could produce warning
      }
      result[i] = func(vval, fval);
    }
  } else {
    Mesh::upwind_func func = fVDDZ;
    DiffLookup *table = UpwindTable;

    if (method != DIFF_DEFAULT) {
      // Lookup function
      func = lookupFunc(table, method);
    }

    stencil fval;
    for (const auto &i : result.region(region)) {
      fval.mm = f[i.offset(0,0,-2)];
      fval.m = f[i.zm()];
      fval.c = f[i];
      fval.p = f[i.zp()];
      fval.pp = f[i.offset(0,0,2)];

      result[i] = func(v[i], fval);
    }
  }

  result.setLocation(diffloc);

#if CHECK > 0
  // Mark boundaries as invalid
  result.bndry_xin = result.bndry_xout = result.bndry_yup = result.bndry_ydown = false;
#endif

  return result;
}

/*******************************************************************************
 * Flux conserving schemes
 *******************************************************************************/

const Field2D Mesh::indexFDDX(const Field2D &v, const Field2D &f, CELL_LOC outloc,
                              DIFF_METHOD method, REGION region) {
  TRACE("Mesh::::indexFDDX(Field2D, Field2D)");

  if ((method == DIFF_SPLIT) || ((method == DIFF_DEFAULT) && (fFDDX == nullptr))) {
    // Split into an upwind and a central differencing part
    // d/dx(v*f) = v*d/dx(f) + f*d/dx(v)
    return indexVDDX(v, f, outloc, DIFF_DEFAULT) + f * indexDDX(v);
  }

  Mesh::flux_func func = fFDDX;
  if (method != DIFF_DEFAULT) {
    // Lookup function
    func = lookupFunc(FluxTable, method);
  }

  Field2D result(this);
  result.allocate(); // Make sure data allocated

  if (StaggerGrids &&
      ((v.getLocation() != CELL_CENTRE) || (f.getLocation() != CELL_CENTRE))) {
    // Staggered differencing
    throw BoutException("Unhandled staggering");
  }
  else {
    result.setLocation(CELL_CENTRE);
  }

  ASSERT1(this == v.getMesh());
  ASSERT1(this == f.getMesh());

  if (this->xstart > 1) {
    // Two or more guard cells

    stencil fs;
    stencil vs;
    for (const auto &i : result.region(region)) {
      fs.c = f[i];
      fs.p = f[i.xp()];
      fs.m = f[i.xm()];
      fs.pp = f[i.offset(2, 0, 0)];
      fs.mm = f[i.offset(-2, 0, 0)];

      vs.c = v[i];
      vs.p = v[i.xp()];
      vs.m = v[i.xm()];
      vs.pp = v[i.offset(2, 0, 0)];
      vs.mm = v[i.offset(-2, 0, 0)];

      result[i] = func(vs, fs);
    }
  } else if (this->xstart == 1) {
    // Only one guard cell

    stencil fs;
    fs.pp = nan("");
    fs.mm = nan("");
    stencil vs;
    vs.pp = nan("");
    vs.mm = nan("");

    for (const auto &i : result.region(region)) {
      fs.c = f[i];
      fs.p = f[i.xp()];
      fs.m = f[i.xm()];

      vs.c = v[i];
      vs.p = v[i.xp()];
      vs.m = v[i.xm()];

      result[i] = func(vs, fs);
    }
  } else {
    // No guard cells
    throw BoutException("Error: Derivatives in X requires at least one guard cell");
  }

#if CHECK > 0
  // Mark boundaries as invalid
  result.bndry_xin = result.bndry_xout = false;
#endif

  return result;
}

const Field3D Mesh::indexFDDX(const Field3D &v, const Field3D &f, CELL_LOC outloc,
                              DIFF_METHOD method, REGION region) {
  TRACE("Mesh::indexFDDX(Field3D, Field3D)");

  if ((method == DIFF_SPLIT) || ((method == DIFF_DEFAULT) && (fFDDX == nullptr))) {
    // Split into an upwind and a central differencing part
    // d/dx(v*f) = v*d/dx(f) + f*d/dx(v)
    return indexVDDX(v, f, outloc, DIFF_DEFAULT) + indexDDX(v, outloc, DIFF_DEFAULT) * f;
  }

  Mesh::flux_func func = fFDDX;
  DiffLookup *table = FluxTable;

  CELL_LOC vloc = v.getLocation();
  CELL_LOC inloc = f.getLocation(); // Input location
  CELL_LOC diffloc = inloc;         // Location of differential result

  if (StaggerGrids && (outloc == CELL_DEFAULT)) {
    // Take care of CELL_DEFAULT case
    outloc = diffloc; // No shift (i.e. same as no stagger case)
  }

  if (StaggerGrids && (vloc != inloc)) {
    // Staggered grids enabled, and velocity at different location to value
    if (vloc == CELL_XLOW) {
      // V staggered w.r.t. variable
      func = sfFDDX;
      table = FluxStagTable;
      diffloc = CELL_CENTRE;
    } else if ((vloc == CELL_CENTRE) && (inloc == CELL_XLOW)) {
      // Shifted
      func = sfFDDX;
      table = FluxStagTable;
      diffloc = CELL_XLOW;
    } else {
      // More complicated. Deciding what to do here isn't straightforward
      throw BoutException("Unhandled shift in indexFDDX");
    }
  }

  if (method != DIFF_DEFAULT) {
    // Lookup function
    func = lookupFunc(table, method);
  }

  ASSERT1(this == f.getMesh());
  ASSERT1(this == v.getMesh());

  Field3D result(this);
  result.allocate(); // Make sure data allocated

  if (this->xstart > 1) {
    // Two or more guard cells
    if (StaggerGrids) {
      if ((vloc == CELL_CENTRE) && (diffloc == CELL_XLOW)) {
        // Producing a stencil centred around a lower X value

        stencil fs, vs;
        vs.c = nan("");
        for (const auto &i : result.region(region)) {
          // Location of f always the same as the output
          fs.c = f[i];
          fs.p = f[i.xp()];
          fs.m = f[i.xm()];
          fs.pp = f[i.offset(2, 0, 0)];
          fs.mm = f[i.offset(-2, 0, 0)];

          // Note: Location in diffloc

          vs.mm = v[i.offset(-2, 0, 0)];
          vs.m = v[i.xm()];
          vs.p = v[i];
          vs.pp = v[i.xp()];

          result[i] = func(vs, fs);
        }
      } else if ((vloc == CELL_XLOW) && (diffloc == CELL_CENTRE)) {
        // Stencil centred around a cell centre
        stencil fs, vs;
        vs.c = nan("");
        for (const auto &i : result.region(region)) {
          // Location of f always the same as the output
          fs.c = f[i];
          fs.p = f[i.xp()];
          fs.m = f[i.xm()];
          fs.pp = f[i.offset(2, 0, 0)];
          fs.mm = f[i.offset(-2, 0, 0)];

          vs.mm = v[i.xm()];
          vs.m = v[i];
          vs.p = v[i.xp()];
          vs.pp = v[i.offset(2, 0, 0)];

          result[i] = func(vs, fs);
        }
      } else {
        throw BoutException("Unhandled staggering");
      }
    } else {
      // Non-staggered, two or more guard cells
      stencil fs;
      stencil vs;
      for (const auto &i : result.region(region)) {
        // Location of f always the same as the output
        fs.c = f[i];
        fs.p = f[i.xp()];
        fs.m = f[i.xm()];
        fs.pp = f[i.offset(2, 0, 0)];
        fs.mm = f[i.offset(-2, 0, 0)];

        // Note: Location in diffloc
        vs.c = v[i];
        vs.p = v[i.xp()];
        vs.m = v[i.xm()];
        vs.pp = v[i.offset(2, 0, 0)];
        vs.mm = v[i.offset(-2, 0, 0)];

        result[i] = func(vs, fs);
      }
    }
  } else if (this->xstart == 1) {
    // One guard cell

    stencil fs;
    fs.pp = nan("");
    fs.mm = nan("");

    stencil vs;
    vs.pp = nan("");
    vs.mm = nan("");
    vs.c = nan("");

    if (StaggerGrids) {
      if ((vloc == CELL_CENTRE) && (diffloc == CELL_XLOW)) {
        // Producing a stencil centred around a lower X value

        for (const auto &i : result.region(region)) {
          // Location of f always the same as the output
          fs.c = f[i];
          fs.p = f[i.xp()];
          fs.m = f[i.xm()];

          // Note: Location in diffloc
          vs.m = v[i.xm()];
          vs.p = v[i];
          vs.pp = v[i.xp()];

          result[i] = func(vs, fs);
        }
      } else if ((vloc == CELL_XLOW) && (diffloc == CELL_CENTRE)) {
        // Stencil centred around a cell centre
        for (const auto &i : result.region(region)) {
          // Location of f always the same as the output
          fs.c = f[i];
          fs.p = f[i.xp()];
          fs.m = f[i.xm()];

          vs.mm = v[i.xm()];
          vs.m = v[i];
          vs.p = v[i.xp()];

          result[i] = func(vs, fs);
        }
      } else {
        throw BoutException("Unhandled staggering");
      }
    } else {
      // Non-staggered, one guard cell
      for (const auto &i : result.region(region)) {
        // Location of f always the same as the output
        fs.c = f[i];
        fs.p = f[i.xp()];
        fs.m = f[i.xm()];

        vs.c = v[i];
        vs.p = v[i.xp()];
        vs.m = v[i.xm()];

        result[i] = func(vs, fs);
      }
    }
  } else {
    // No guard cells
    throw BoutException("Error: Derivatives in X requires at least one guard cell");
  }

  result.setLocation(diffloc);

#if CHECK > 0
  // Mark boundaries as invalid
  result.bndry_xin = result.bndry_xout = result.bndry_yup = result.bndry_ydown = false;
#endif

  return result;
}

/////////////////////////////////////////////////////////////////////////

const Field2D Mesh::indexFDDY(const Field2D &v, const Field2D &f, CELL_LOC outloc,
                              DIFF_METHOD method, REGION region) {
  TRACE("Mesh::indexFDDY(Field2D, Field2D)");

  ASSERT1(this == v.getMesh());
  ASSERT1(this == f.getMesh());

  CELL_LOC diffloc = f.getLocation();

  if ((method == DIFF_SPLIT) || ((method == DIFF_DEFAULT) && (fFDDY == nullptr))) {
    // Split into an upwind and a central differencing part
    // d/dx(v*f) = v*d/dx(f) + f*d/dx(v)
    return indexVDDY(v, f, outloc, DIFF_DEFAULT) + f * indexDDY(v);
  }

  Mesh::flux_func func = fFDDY;
  if (method != DIFF_DEFAULT) {
    // Lookup function
    func = lookupFunc(FluxTable, method);
  }

  Field2D result(this);
  result.allocate(); // Make sure data allocated

  if (StaggerGrids &&
      ((v.getLocation() != CELL_CENTRE) || (f.getLocation() != CELL_CENTRE))) {
    // Staggered differencing
    throw BoutException("Unhandled staggering");
  }
  else {
    result.setLocation(CELL_CENTRE);
  }

  if (this->ystart > 1) {
    // Two or more guard cells
    stencil fs, vs;
    for (const auto &i : result.region(region)) {

      fs.c = f[i];
      fs.p = f[i.yp()];
      fs.m = f[i.ym()];
      fs.pp = f[i.offset(0, 2, 0)];
      fs.mm = f[i.offset(0, -2, 0)];

      vs.c = v[i];
      vs.p = v[i.yp()];
      vs.m = v[i.ym()];
      vs.pp = v[i.offset(0, 2, 0)];
      vs.mm = v[i.offset(0, -2, 0)];

      result[i] = func(vs, fs);
    }

  } else if (this->ystart == 1) {
    // Only one guard cell

    stencil fs;
    fs.pp = nan("");
    fs.mm = nan("");
    stencil vs;
    vs.pp = nan("");
    vs.mm = nan("");

    for (const auto &i : result.region(region)) {
      fs.c = f[i];
      fs.p = f[i.yp()];
      fs.m = f[i.ym()];

      vs.c = v[i];
      vs.p = v[i.yp()];
      vs.m = v[i.ym()];
      result[i] = func(vs, fs);
    }
  } else {
    // No guard cells
    throw BoutException("Error: Derivatives in Y requires at least one guard cell");
  }

  result.setLocation(diffloc);

#if CHECK > 0
  // Mark boundaries as invalid
  result.bndry_xin = result.bndry_xout = false;
#endif

  return result;
}

const Field3D Mesh::indexFDDY(const Field3D &v, const Field3D &f, CELL_LOC outloc,
                              DIFF_METHOD method, REGION region) {
  TRACE("Mesh::indexFDDY");

  if ((method == DIFF_SPLIT) || ((method == DIFF_DEFAULT) && (fFDDY == nullptr))) {
    // Split into an upwind and a central differencing part
    // d/dx(v*f) = v*d/dx(f) + f*d/dx(v)
    return indexVDDY(v, f, outloc, DIFF_DEFAULT) + indexDDY(v, outloc, DIFF_DEFAULT) * f;
  }
  Mesh::flux_func func = fFDDY;
  DiffLookup *table = FluxTable;

  CELL_LOC vloc = v.getLocation();
  CELL_LOC inloc = f.getLocation(); // Input location
  CELL_LOC diffloc = inloc;         // Location of differential result

  if (StaggerGrids && (outloc == CELL_DEFAULT)) {
    // Take care of CELL_DEFAULT case
    outloc = diffloc; // No shift (i.e. same as no stagger case)
  }

  if (StaggerGrids && (vloc != inloc)) {
    // Staggered grids enabled, and velocity at different location to value
    if (vloc == CELL_YLOW) {
      // V staggered w.r.t. variable
      func = sfFDDY;
      table = FluxStagTable;
      diffloc = CELL_CENTRE;
    } else if ((vloc == CELL_CENTRE) && (inloc == CELL_YLOW)) {
      // Shifted
      func = sfFDDY;
      table = FluxStagTable;
      diffloc = CELL_YLOW;
    } else {
      // More complicated. Deciding what to do here isn't straightforward
      throw BoutException("Unhandled shift in indexFDDY");
    }
  }

  if (method != DIFF_DEFAULT) {
    // Lookup function
    func = lookupFunc(table, method);
  }

  if (func == nullptr) {
    // To catch when no function
    return indexVDDY(v, f, outloc, DIFF_DEFAULT) + indexDDY(v, outloc, DIFF_DEFAULT) * f;
  }

  ASSERT1(this == v.getMesh());
  ASSERT1(this == f.getMesh());

  Field3D result(this);
  result.allocate(); // Make sure data allocated

  // There are four cases, corresponding to whether or not f and v
  // have yup, ydown fields.

  // If vUseUpDown is true, field "v" has distinct yup and ydown fields which
  // will be used to calculate a derivative along
  // the magnetic field
  bool vUseUpDown = (v.hasYupYdown() && ((&v.yup() != &v) || (&v.ydown() != &v)));
  bool fUseUpDown = (f.hasYupYdown() && ((&f.yup() != &f) || (&f.ydown() != &f)));

  if (vUseUpDown && fUseUpDown) {
    // Both v and f have up/down fields
    stencil vval, fval;
    vval.mm = nan("");
    vval.pp = nan("");
    fval.mm = nan("");
    fval.pp = nan("");
    for (const auto &i : result.region(region)) {

      fval.m = f.ydown()[i.ym()];
      fval.c = f[i];
      fval.p = f.yup()[i.yp()];

      vval.m = v.ydown()[i.ym()];
      vval.c = v[i];
      vval.p = v.yup()[i.yp()];

      if(StaggerGrids && (diffloc != CELL_DEFAULT) && (diffloc != vloc)) {
        // Non-centred stencil
        if((vloc == CELL_CENTRE) && (diffloc == CELL_YLOW)) {
          // Producing a stencil centred around a lower Y value
          vval.pp = vval.p;
          vval.p  = vval.c;
        }else if(vloc == CELL_YLOW) {
          // Stencil centred around a cell centre
          vval.mm = vval.m;
          vval.m  = vval.c;
        }
        // Shifted in one direction -> shift in another
        // Could produce warning
      }
      result[i] = func(vval, fval);
    }
  }
  else if (vUseUpDown) {
    // Only v has up/down fields
    // f must shift to field aligned coordinates
    Field3D f_fa = this->toFieldAligned(f);

    stencil vval;
    vval.mm = nan("");
    vval.pp = nan("");

    stencil fval;
    for (const auto &i : result.region(region)) {

      fval.mm = f_fa[i.offset(0, -2, 0)];
      fval.m = f_fa[i.ym()];
      fval.c = f_fa[i];
      fval.p = f_fa[i.yp()];
      fval.pp = f_fa[i.offset(0, 2, 0)];

      vval.m = v.ydown()[i.ym()];
      vval.c = v[i];
      vval.p = v.yup()[i.yp()];

      if(StaggerGrids && (diffloc != CELL_DEFAULT) && (diffloc != vloc)) {
        // Non-centred stencil
        if((vloc == CELL_CENTRE) && (diffloc == CELL_YLOW)) {
          // Producing a stencil centred around a lower Y value
          vval.pp = vval.p;
          vval.p  = vval.c;
        }else if(vloc == CELL_YLOW) {
          // Stencil centred around a cell centre
          vval.mm = vval.m;
          vval.m  = vval.c;
        }
        // Shifted in one direction -> shift in another
        // Could produce warning
      }
      result[i] = func(vval, fval);
    }
  }
  else if (fUseUpDown) {
    // Only f has up/down fields
    // v must shift to field aligned coordinates
    Field3D v_fa = this->toFieldAligned(v);

    stencil vval;

    stencil fval;
    fval.pp = nan("");
    fval.mm = nan("");

    for (const auto &i : result.region(region)) {

      fval.m = f.ydown()[i.ym()];
      fval.c = f[i];
      fval.p = f.yup()[i.yp()];

      vval.mm = v_fa[i.offset(0,-2,0)];
      vval.m = v_fa[i.ym()];
      vval.c = v_fa[i];
      vval.p = v_fa[i.yp()];
      vval.pp = v_fa[i.offset(0,2,0)];

      if(StaggerGrids && (diffloc != CELL_DEFAULT) && (diffloc != vloc)) {
        // Non-centred stencil
        if((vloc == CELL_CENTRE) && (diffloc == CELL_YLOW)) {
          // Producing a stencil centred around a lower Y value
          vval.pp = vval.p;
          vval.p  = vval.c;
        }else if(vloc == CELL_YLOW) {
          // Stencil centred around a cell centre
          vval.mm = vval.m;
          vval.m  = vval.c;
        }
        // Shifted in one direction -> shift in another
        // Could produce warning
      }
      result[i] = func(vval, fval);
    }
  }
  else {
    // Both must shift to field aligned
    Field3D v_fa = this->toFieldAligned(v);
    Field3D f_fa = this->toFieldAligned(f);

    stencil vval, fval;

    for (const auto &i : result.region(region)) {

      fval.mm = f_fa[i.offset(0,-2,0)];
      fval.m = f_fa[i.ym()];
      fval.c = f_fa[i];
      fval.p = f_fa[i.yp()];
      fval.pp = f_fa[i.offset(0,2,0)];

      vval.mm = v_fa[i.offset(0,-2,0)];
      vval.m = v_fa[i.ym()];
      vval.c = v_fa[i];
      vval.p = v_fa[i.yp()];
      vval.pp = v_fa[i.offset(0,2,0)];

      if(StaggerGrids && (diffloc != CELL_DEFAULT) && (diffloc != vloc)) {
        // Non-centred stencil
        if((vloc == CELL_CENTRE) && (diffloc == CELL_YLOW)) {
          // Producing a stencil centred around a lower Y value
          vval.pp = vval.p;
          vval.p  = vval.c;
        }else if(vloc == CELL_YLOW) {
          // Stencil centred around a cell centre
          vval.mm = vval.m;
          vval.m  = vval.c;
        }
        // Shifted in one direction -> shift in another
        // Could produce warning
      }
      result[i] = func(vval, fval);
    }
  }

  result.setLocation(diffloc);

#if CHECK > 0
  // Mark boundaries as invalid
  result.bndry_xin = result.bndry_xout = result.bndry_yup = result.bndry_ydown = false;
#endif

  return result;
}

/////////////////////////////////////////////////////////////////////////

const Field3D Mesh::indexFDDZ(const Field3D &v, const Field3D &f, CELL_LOC outloc,
                              DIFF_METHOD method, REGION region) {
  TRACE("Mesh::indexFDDZ(Field3D, Field3D)");
  if ((method == DIFF_SPLIT) || ((method == DIFF_DEFAULT) && (fFDDZ == nullptr))) {
    // Split into an upwind and a central differencing part
    // d/dx(v*f) = v*d/dx(f) + f*d/dx(v)
    return indexVDDZ(v, f, outloc, DIFF_DEFAULT) +
           indexDDZ(v, outloc, DIFF_DEFAULT, true) * f;
  }

  Mesh::flux_func func = fFDDZ;
  DiffLookup *table = FluxTable;

  CELL_LOC vloc = v.getLocation();
  CELL_LOC inloc = f.getLocation(); // Input location
  CELL_LOC diffloc = inloc;         // Location of differential result

  if (StaggerGrids && (outloc == CELL_DEFAULT)) {
    // Take care of CELL_DEFAULT case
    outloc = diffloc; // No shift (i.e. same as no stagger case)
  }

  if (StaggerGrids && (vloc != inloc)) {
    // Staggered grids enabled, and velocity at different location to value
    if (vloc == CELL_ZLOW) {
      // V staggered w.r.t. variable
      func = sfFDDZ;
      table = FluxStagTable;
      diffloc = CELL_CENTRE;
    } else if ((vloc == CELL_CENTRE) && (inloc == CELL_ZLOW)) {
      // Shifted
      func = sfFDDZ;
      table = FluxStagTable;
      diffloc = CELL_ZLOW;
    } else {
      // More complicated. Deciding what to do here isn't straightforward
      throw BoutException("Unhandled shift in indexFDDZ");
    }
  }

  if (method != DIFF_DEFAULT) {
    // Lookup function
    func = lookupFunc(table, method);
  }

  ASSERT1(this == v.getMesh());
  ASSERT1(this == f.getMesh());

  Field3D result(this);
  result.allocate(); // Make sure data allocated

  stencil vval, fval;
  for (const auto &i : result.region(region)) {

    fval.mm = f[i.offset(0,0,-2)];
    fval.m = f[i.zm()];
    fval.c = f[i];
    fval.p = f[i.zp()];
    fval.pp = f[i.offset(0,0,2)];

    vval.mm = v[i.offset(0,0,-2)];
    vval.m = v[i.zm()];
    vval.c = v[i];
    vval.p = v[i.zp()];
    vval.pp = v[i.offset(0,0,2)];

    if(StaggerGrids && (diffloc != CELL_DEFAULT) && (diffloc != vloc)) {
      // Non-centred stencil

      if((vloc == CELL_CENTRE) && (diffloc == CELL_ZLOW)) {
      // Producing a stencil centred around a lower Z value
        vval.pp = vval.p;
        vval.p  = vval.c;

      }else if(vloc == CELL_ZLOW) {
        // Stencil centred around a cell centre

        vval.mm = vval.m;
        vval.m  = vval.c;
      }
      // Shifted in one direction -> shift in another
      // Could produce warning
    }
    result[i] = func(vval, fval);
  }

  result.setLocation(diffloc);

#if CHECK > 0
  // Mark boundaries as invalid
  result.bndry_xin = result.bndry_xout = result.bndry_yup = result.bndry_ydown = false;
#endif

  return result;
}
