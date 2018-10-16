
class BoundaryOp;
class BoundaryModifier;

#ifndef __BNDRY_OP__
#define __BNDRY_OP__

#include "boundary_region.hxx"
#include "field2d.hxx"
#include "field3d.hxx"
#include "vector2d.hxx"
#include "vector3d.hxx"
#include "unused.hxx"

#include <cmath>
#include <string>
#include <list>
using std::string;
using std::list;

/// An operation on a boundary
class BoundaryOp {
public:
  BoundaryOp() {
    bndry = nullptr;
    apply_to_ddt = false;
  }
  BoundaryOp(BoundaryRegion *region) {bndry = region; apply_to_ddt=false;}
  virtual ~BoundaryOp() {}

  // Note: All methods must implement clone, except for modifiers (see below)
  virtual BoundaryOp* clone(BoundaryRegion *UNUSED(region), const list<string> &UNUSED(args)) {
    return nullptr;
  }

  /// Apply a boundary condition on field f
  virtual void apply(Field2D &f,BoutReal t = 0.) = 0;
  virtual void apply(Field3D &f,BoutReal t = 0.) = 0;

  virtual void apply(Vector2D &f) {
    apply(f.x);
    apply(f.y);
    apply(f.z);
  }

  virtual void apply(Vector3D &f) {
    apply(f.x);
    apply(f.y);
    apply(f.z);
  }

  /// Apply a boundary condition on ddt(f)
  virtual void apply_ddt(Field2D &f);
  virtual void apply_ddt(Field3D &f);
  virtual void apply_ddt(Vector2D &f) {
    apply(ddt(f));
  }
  virtual void apply_ddt(Vector3D &f) {
    apply(ddt(f));
  }

  BoundaryRegion *bndry;
  bool apply_to_ddt; // True if this boundary condition should be applied on the time derivatives, false if it should be applied to the field values

protected:
  //virtual void apply(Field3D &f, BoutReal t = 0.) override;
  //virtual void apply(Field2D &f, BoutReal t = 0.) override;

  // Apply boundary condition at a point
  virtual void applyAtPoint(Field3D &f, BoutReal val, int x, int bx, int y, int by, int z, Coordinates* metric) = 0;
  virtual void applyAtPoint(Field2D &f, BoutReal val, int x, int bx, int y, int by, int z, Coordinates* metric) = 0;

  // Apply to staggered grid
  virtual void applyAtPointStaggered(Field3D &f, BoutReal val, int x, int bx, int y, int by, int z, Coordinates* metric) = 0;
  virtual void applyAtPointStaggered(Field2D &f, BoutReal val, int x, int bx, int y, int by, int z, Coordinates* metric) = 0;

  // extrapolate to further guard cells
  virtual void extrapFurther(Field3D &f, int x, int bx, int y, int by, int z);
  virtual void extrapFurther(Field2D &f, int x, int bx, int y, int by, int z);
};

class BoundaryModifier : public BoundaryOp {
public:
  BoundaryModifier() : op(nullptr) {}
  BoundaryModifier(BoundaryOp *operation) : BoundaryOp(operation->bndry), op(operation) {}
  virtual BoundaryOp* cloneMod(BoundaryOp *op, const list<string> &args) = 0;
protected:
  BoundaryOp *op;
};

#endif // __BNDRY_OP__
