#include <cmath>
#define _CRT_SECURE_NO_WARNINGS 1
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <algorithm>
#include <iostream>

#define PI 3.1415926535
#define G_FORCE 9.81

#include "lbfgs.c"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <list>
#include <sstream>
#include <stdio.h>
#include <string>

#define DEBUG 0

//////////// Vector class as in LAB1 //////////////
class Vector {
  public:
    explicit Vector(double x = 0, double y = 0, double z = 0) {
        data[0] = x;
        data[1] = y;
        data[2] = z;
    }
    double norm2() const {
        return data[0] * data[0] + data[1] * data[1] + data[2] * data[2];
    }
    double norm() const {
        return sqrt(norm2());
    }
    void normalize() {
        double n = norm();
        data[0] /= n;
        data[1] /= n;
        data[2] /= n;
    }
    Vector cross(const Vector &b) const {
        return Vector(data[1] * b[2] - data[2] * b[1],
                      data[2] * b[0] - data[0] * b[2],
                      data[0] * b[1] - data[1] * b[0]);
    }
    double operator[](int i) const { return data[i]; };
    double &operator[](int i) { return data[i]; };
    double data[3];
};

Vector operator+(const Vector &a, const Vector &b) {
    return Vector(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}
Vector operator-(const Vector &a, const Vector &b) {
    return Vector(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}
Vector operator-(const Vector &a) {
    return Vector(-a[0], -a[1], -a[2]);
}
Vector operator*(const double a, const Vector &b) {
    return Vector(a * b[0], a * b[1], a * b[2]);
}
Vector operator*(const Vector &a, const double b) {
    return Vector(a[0] * b, a[1] * b, a[2] * b);
}
Vector operator*(const Vector &a, const Vector &b) {
    return Vector(a[0] * b[0], a[1] * b[1], a[2] * b[2]);
}
Vector operator/(const Vector &a, const double b) {
    return Vector(a[0] / b, a[1] / b, a[2] / b);
}
double dot(const Vector &a, const Vector &b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Vector cross(const Vector &a, const Vector &b) {
    return Vector(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
}

////////// end VECTOR CLASS /////////////////////////////////////////

// KNN for KNN optimization
std::list<Vector> knn(const Vector &point, const std::vector<Vector> &points, size_t k) {
    std::list<Vector> nearest_neighbors;
    double distance = 0.;
    bool insert;
    for (Vector neighbor : points) {
        insert = false;
        distance = (point - neighbor).norm2();
        std::list<Vector>::iterator it = nearest_neighbors.begin();
        for (; it != nearest_neighbors.end(); it++) {
            if (distance > (point - *it).norm2()) {
                break;
            }
            insert = true;
        }
        if (insert) {
            nearest_neighbors.insert(it, neighbor);
            if (nearest_neighbors.size() > k) {
                nearest_neighbors.pop_front();
            }
        } else if (nearest_neighbors.size() < k) {
            nearest_neighbors.push_front(neighbor);
        }
    }
    return nearest_neighbors;
}

//////////////////// POLYGON CLASS /////////////////////////////

class Polygon {
  public:
    std::vector<Vector> vertices;
    Polygon() {
        vertices = std::vector<Vector>();
    };
    void add_vertex(Vector vertex) {
        vertices.push_back(vertex);
    };
    double area() {
        if (vertices.size() == 0) {
#if DEBUG
            std::cout << "ILLEGAL NB OF VERTICES FOR AREA: " << vertices.size() << std::endl;
#endif
            return 0.0;
        } else if (vertices.size() < 3) {
            std::cout << "ILLEGAL NB OF VERTICES FOR AREA: " << vertices.size() << std::endl;
            throw std::invalid_argument("1 or 2 vertices not enough to compute area");
        }
        // We triangulate
        double res = 0.;
        Vector origin = vertices[0];

        // for each triangle
        for (size_t i = 1; i < vertices.size() - 1; i++) {
            std::vector<Vector> T = {origin, vertices[i], vertices[i + 1]}; // this triangulation works since Laguerre cells are always convex
            // Add triangle's area to result
            res += 0.5 * std::abs(T[0][0] * (T[1][1] - T[2][1]) + T[1][0] * (T[2][1] - T[0][1]) + T[2][0] * (T[0][1] - T[1][1]));
        }
        return res;
    }

    Vector centroid() {
        Vector res = Vector(0, 0);
        for (Vector vertex : vertices) {
            res = res + vertex;
        }
        return res / vertices.size();
    }
    

    double integral_norm_diff(Vector P_i) {
        if (vertices.size() == 0) {
            return 0.;
        } else if (vertices.size() < 3) {
            std::cout << "NOT ENOUGH VERTICES FOR AREA COMPUTATION";
            throw std::invalid_argument("1 or 2 vertices not enough to compute area");
        }
        double res = 0.;
        // We triangulate
        Vector origin = vertices[0];
        size_t j = 1;
        // for each triangle
        for (; j < this->vertices.size() - 1; j++) {
            std::vector<Vector> T = {origin, vertices[j], vertices[j + 1]}; // this triangulation works since Laguerre cells are always convex
            for (int l = 0; l < 3; l++) {
                for (int k = 0; k <= l; k++) {
                    double step_res = std::abs(dot(T[k] - P_i, T[l] - P_i)) / 2;
                    res += step_res;
                }
            }
        }
        return res;
    }

    double integral_norm_diff2(Vector P_i) {
        if (vertices.size() == 0) {
            return 0.;
        } else if (vertices.size() < 3) {
            std::cout << "NOT ENOUGH VERTICES FOR AREA COMPUTATION";
            throw std::invalid_argument("1 or 2 vertices not enough to compute area");
        }
        double res = 0.;
        for (size_t k = 0; k < vertices.size(); k++) {
            double X_k = vertices[k][0];
            double Y_k = vertices[k][1];
            double X_k_1 = vertices[(k > 0) ? (k - 1) : vertices.size() - 1][0];
            double Y_k_1 = vertices[(k > 0) ? (k - 1) : vertices.size() - 1][1];
            res += (X_k_1 * Y_k - X_k * Y_k_1) *
                   (pow(X_k_1, 2) + X_k_1 * X_k + pow(X_k, 2) + pow(Y_k_1, 2) + Y_k_1 * Y_k + pow(Y_k, 2) - 4. * (P_i[0] * (X_k_1 + X_k) + P_i[1] * (Y_k_1 + Y_k)) + 6. * P_i.norm2());
        }
        return std::abs(res) / 12.;
    }
};

Polygon regular_n_gon(int n, double r, Vector center = Vector(0, 0)) {
    Polygon points;
    for (int i = 0; i < n; i++) {
        points.add_vertex(center + Vector(r * cos(2 * PI * i / n), r * sin(2 * PI * i / n)));
    }
    return points;
}
std::vector<Polygon> points_to_square(std::vector<Vector> points) {
    std::vector<Polygon> squares;
    for (Vector point : points) {
        Polygon square = Polygon();
        square.add_vertex(point + Vector(-0.001, -0.001));
        square.add_vertex(point + Vector(-0.001, 0.001));
        square.add_vertex(point + Vector(0.001, 0.001));
        square.add_vertex(point + Vector(0.001, -0.001));
        squares.push_back(square);
    }
    return squares;
}

void print(Polygon &p) {
    for (size_t i = 0; i < p.vertices.size(); i++) {
        std::cout << p.vertices[i][0] << " " << p.vertices[i][1] << std::endl;
    }
}

//////////////////// end POLYGON /////////////////////////////

///////////////// Polygon cliping //////////////////////

// intersection of the finite edge [A,B] with the line (u,v)
Vector intersect(Vector A, Vector B, Vector u, Vector v) {
    Vector N = Vector(v[1] - u[1], u[0] - v[0]);
    double t = dot(u - A, N) / dot(B - A, N);
    Vector P = A + t * (B - A);

    return P;
}

bool inside(Vector P, Vector u, Vector v) {
    Vector N = Vector(v[1] - u[1], u[0] - v[0]);
    Vector V = P - u;
    return dot(N, V) > 0;
}

// Sutherland-Hodgman polygon clipping algorithm (●'◡'●)
void clip(Polygon &subjectPolygon, Polygon &clipPolygon) {
    // clipPolygon must be convex!
    for (int edge = 0; edge < (int)clipPolygon.vertices.size(); edge++) {


        Vector b = clipPolygon.vertices[edge];
        Vector a = clipPolygon.vertices[(edge - 1 + clipPolygon.vertices.size()) % clipPolygon.vertices.size()];

        Polygon outPolygon = Polygon();

        for (int i = 0; i < (int)subjectPolygon.vertices.size(); i++) {

            Vector curVertex = subjectPolygon.vertices[i];
            Vector prevVertex = subjectPolygon.vertices[(i - 1 + subjectPolygon.vertices.size()) % subjectPolygon.vertices.size()];
            Vector intersection = intersect(prevVertex, curVertex, a, b);

            if (inside(curVertex, a, b)) {
                if (!inside(prevVertex, a, b)) {
                    // The subject polygon edge crosses the clip edge, and we leave the clipping area
                    outPolygon.add_vertex(intersection);
                }
                outPolygon.add_vertex(curVertex);

            } else if (inside(prevVertex, a, b)) {
                // The subject polygon edge crosses the clip edge, and we enter the clipping area
                outPolygon.add_vertex(intersection);
            }
        }
        subjectPolygon = outPolygon;
    }
}

std::vector<Polygon> voronoi_parallel_linear_enumeration(std::vector<Vector> points, std::vector<double> weights = std::vector<double>(), bool crop_circle = false, double air = 0.0) {
    bool power = weights.size() == points.size() ? true : false;
    size_t n = points.size();

    // add extra dimension
    double m = 0.;
    if (power) {
        m = *std::max_element(weights.begin(), weights.end());
        for (size_t i = 0; i < points.size(); i++) {
            points[i][2] = std::sqrt(m - weights[i]);
#if DEBUG
            std::cout << "W[" << i << "] = " << weights[i] << ", ";
#endif
        }
    }
#if DEBUG
    if (crop_circle) {
        std::cout << " air = " << air << std::endl;
    }
    std::cout << "\n";
#endif
    std::vector<Vector> &points_prime = points;

    // find max dimensions to create large enough polynomials
    Vector max = Vector(0, 0);
    for (Vector point : points) {
        if (std::abs(point[0]) > max[0]) {
            max[0] = std::abs(point[0]);
        }
        if (std::abs(point[1]) > max[1]) {
            max[1] = std::abs(point[1]);
        }
    }
    max[0] *= 2;
    max[1] *= 2;

    const double max_distance = max.norm();

    // for each point, create the big polynomial and cut bissectors
    std::vector<Polygon> results(points.size());
    std::vector<Polygon> square_pts = points_to_square(points);
    Polygon boundaries = Polygon();

    boundaries.add_vertex(Vector(0, 0));
    boundaries.add_vertex(Vector(0, 1));
    boundaries.add_vertex(Vector(1, 1));
    boundaries.add_vertex(Vector(1, 0));

    #pragma omp parallel for
    for (size_t ppindex = 0; ppindex < points_prime.size(); ppindex++) {
        Vector point_prime = points_prime[ppindex];
        Vector point = Vector(point_prime[0], point_prime[1]);
        size_t k = 7; //points.size(); // to fine-tune
        bool done = false;

        while (!done) {

            // create big square
            // Polygon big_square = Polygon();
            // big_square.add_vertex(point + Vector(-max[0], -max[1]));
            // big_square.add_vertex(point + Vector(-max[0], max[1]));
            // big_square.add_vertex(point + Vector(max[0], max[1]));
            // big_square.add_vertex(point + Vector(max[0], -max[1]));
            Polygon big_square = boundaries;

            std::vector<Polygon> tmp = square_pts;

            // find k nearest neighbors
            std::list<Vector> nearest_neighbors = knn(point_prime, points_prime, k);

            // for each neighbor: cut bissector
            // Create bissector square
            
            double longest_dist = std::numeric_limits<double>::infinity();
            for (std::list<Vector>::iterator it = std::prev(std::prev(nearest_neighbors.end())); it != std::prev(nearest_neighbors.begin()); it--) {
                Vector p2_prime = *it;
                Vector p2 = Vector(p2_prime[0], p2_prime[1]);
                if ((p2_prime - point_prime).norm() > 2 * longest_dist) {
                    done = true;
                    break;
                }
                Polygon bissector_square = Polygon();
                Vector midpoint = point + (p2 - point) / 2;
                Vector bissector_vector = Vector(midpoint[1] - point[1], point[0] - midpoint[0]);
                Vector ortho_bissector_vector = midpoint - point;
                ortho_bissector_vector.normalize();
                bissector_vector.normalize();

                if (power) {
                    double w1 = -(pow(point_prime[2], 2) - m);
                    double w2 = -(pow(p2_prime[2], 2) - m);
                    // std::cout << w1 << "  " << w2 << std::endl;
                    Vector correction = ((w1 - w2) / (2 * (p2 - point).norm2())) * (p2 - point);
                    // std::cout << correction[0] << "  " << correction[1] << std::endl;
                    midpoint = midpoint + correction;
                }

                bissector_square.add_vertex(midpoint + bissector_vector * 2 * max_distance);
                bissector_square.add_vertex(point + (-ortho_bissector_vector + bissector_vector) * 2 * max_distance);
                bissector_square.add_vertex(point + (-ortho_bissector_vector - bissector_vector) * 2 * max_distance);
                bissector_square.add_vertex(midpoint - bissector_vector * 2 * max_distance);

                clip(big_square, bissector_square);

                // recomputing best distance
                double new_longest_dist = 0;
                for (Vector vertex : big_square.vertices) {
                    double dist = (point_prime - vertex).norm();
                    if (dist > new_longest_dist) {
                        new_longest_dist = dist;
                    }
                }
                longest_dist = new_longest_dist;
            }
            if (done || k == points.size()) {
                done = true;
                if (crop_circle) {
                    if (!power) {
                        std::cout << "ERROR: crop_circle requires power diagram" << std::endl;
                        throw std::invalid_argument("crop_circle requires power diagram");
                    }
                    Polygon circle = regular_n_gon(100, sqrt(std::max(weights[ppindex] - air, 0.)), point);
                    clip(circle, big_square);
                    big_square = circle;
                }
                results[ppindex] = big_square;
            } else {
                k = std::min(2 * k, points.size());
            }
        }
    }
    //  clean after yourself
    if (power) {
        for (size_t i = 0; i < points.size(); i++) {
            points[i][2] = 0;
        }
    }
    return results;
}

void linear_enumeration(std::vector<Vector> points) {
    for (int i = 0; i < points.size(); i++) {
        std::cout << points[i][0] << " " << points[i][1] << std::endl;
    }
}

////////////////// end Polygon clipping //////////////////////

//////////////////////////// FILE SAVING ////////////////////////////////

// saves a static svg file. The polygon vertices are supposed to be in the range [0..1], and a canvas of size 1000x1000 is created
void save_svg(const std::vector<Polygon> &polygons, std::string filename, std::string fillcol = "none") {
    FILE *f = fopen(filename.c_str(), "w+");
    fprintf(f, "<svg xmlns = \"http://www.w3.org/2000/svg\" width = \"1000\" height = \"1000\">\n");
    for (int i = 0; i < polygons.size(); i++) {
        fprintf(f, "<g>\n");
        fprintf(f, "<polygon points = \"");
        for (int j = 0; j < polygons[i].vertices.size(); j++) {
            fprintf(f, "%3.3f, %3.3f ", (polygons[i].vertices[j][0] * 1000), (1000 - polygons[i].vertices[j][1] * 1000));
        }
        fprintf(f, "\"\nfill = \"%s\" stroke = \"black\"/>\n", fillcol.c_str());
        fprintf(f, "</g>\n");
    }
    fprintf(f, "</svg>\n");
    fclose(f);
}

void save_polygon(Polygon polygon, std::string filename) {
    std::vector<Polygon> polygons;
    polygons.push_back(polygon);
    save_svg(polygons, filename);
}

void save_frame(const std::vector<Polygon> &cells, std::string filename, int frameid = 0) {
    size_t N = cells.size();
    int W = 1000, H = 1000;
    std::vector<unsigned char> image(W * H * 3, 255);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < cells.size(); i++) {

        double bminx = 1E9, bminy = 1E9, bmaxx = -1E9, bmaxy = -1E9;
        for (int j = 0; j < cells[i].vertices.size(); j++) {
            bminx = std::min(bminx, cells[i].vertices[j][0]);
            bminy = std::min(bminy, cells[i].vertices[j][1]);
            bmaxx = std::max(bmaxx, cells[i].vertices[j][0]);
            bmaxy = std::max(bmaxy, cells[i].vertices[j][1]);
        }
        bminx = std::min(W - 1., std::max(0., W * bminx));
        bminy = std::min(H - 1., std::max(0., H * bminy));
        bmaxx = std::max(W - 1., std::max(0., W * bmaxx));
        bmaxy = std::max(H - 1., std::max(0., H * bmaxy));

        for (int y = bminy; y < bmaxy; y++) {
            for (int x = bminx; x < bmaxx; x++) {
                int prevSign = 0;
                bool isInside = true;
                double mindistEdge = 1E9;
                for (int j = 0; j < cells[i].vertices.size(); j++) {
                    double x0 = cells[i].vertices[j][0] * W;
                    double y0 = cells[i].vertices[j][1] * H;
                    double x1 = cells[i].vertices[(j + 1) % cells[i].vertices.size()][0] * W;
                    double y1 = cells[i].vertices[(j + 1) % cells[i].vertices.size()][1] * H;
                    double det = (x - x0) * (y1 - y0) - (y - y0) * (x1 - x0);
                    int sign = det > 0 ? 1 : (det < 0 ? -1 : 0);
                    if (prevSign == 0)
                        prevSign = sign;
                    else if (sign == 0)
                        sign = prevSign;
                    else if (sign != prevSign) {
                        isInside = false;
                        break;
                    }
                    prevSign = sign;
                    double edgeLen = sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0));
                    double distEdge = std::abs(det) / edgeLen;
                    double dotp = (x - x0) * (x1 - x0) + (y - y0) * (y1 - y0);
                    if (dotp < 0 || dotp > edgeLen * edgeLen)
                        distEdge = 1E9;
                    mindistEdge = std::min(mindistEdge, distEdge);
                }
                if (isInside) {
                    if (i < N) { // the N first particles may represent fluid, displayed in blue
                        image[((H - y - 1) * W + x) * 3] = 0;
                        image[((H - y - 1) * W + x) * 3 + 1] = 0;
                        image[((H - y - 1) * W + x) * 3 + 2] = 255;
                    }
                    if (mindistEdge <= 2) {
                        image[((H - y - 1) * W + x) * 3] = 0;
                        image[((H - y - 1) * W + x) * 3 + 1] = 0;
                        image[((H - y - 1) * W + x) * 3 + 2] = 0;
                    }
                }
            }
        }
    }
    std::ostringstream os;
    os << filename << frameid << ".png";
    stbi_write_png(os.str().c_str(), W, H, 3, &image[0], 0);
}

// Adds one frame of an animated svg file. frameid is the frame number (between 0 and nbframes-1).
// polygons is a list of polygons, describing the current frame.
// The polygon vertices are supposed to be in the range [0..1], and a canvas of size 1000x1000 is created
void save_svg_animated(const std::vector<Polygon> &polygons, std::string filename, int frameid, int nbframes) {
    FILE *f;
    if (frameid == 0) {
        f = fopen(filename.c_str(), "w+");
        fprintf(f, "<svg xmlns = \"http://www.w3.org/2000/svg\" width = \"1000\" height = \"1000\">\n");
        fprintf(f, "<g>\n");
    } else {
        f = fopen(filename.c_str(), "a+");
    }
    fprintf(f, "<g>\n");
    for (int i = 0; i < polygons.size(); i++) {
        fprintf(f, "<polygon points = \"");
        for (int j = 0; j < polygons[i].vertices.size(); j++) {
            fprintf(f, "%3.3f, %3.3f ", (polygons[i].vertices[j][0] * 1000), (1000 - polygons[i].vertices[j][1] * 1000));
        }
        fprintf(f, "\"\nfill = \"none\" stroke = \"black\"/>\n");
    }
    fprintf(f, "<animate\n");
    fprintf(f, "	id = \"frame%u\"\n", frameid);
    fprintf(f, "	attributeName = \"display\"\n");
    fprintf(f, "	values = \"");
    for (int j = 0; j < nbframes; j++) {
        if (frameid == j) {
            fprintf(f, "inline");
        } else {
            fprintf(f, "none");
        }
        fprintf(f, ";");
    }
    fprintf(f, "none\"\n	keyTimes = \"");
    for (int j = 0; j < nbframes; j++) {
        fprintf(f, "%2.3f", j / (double)(nbframes));
        fprintf(f, ";");
    }
    fprintf(f, "1\"\n	dur = \"5s\"\n");
    fprintf(f, "	begin = \"0s\"\n");
    fprintf(f, "	repeatCount = \"indefinite\"/>\n");
    fprintf(f, "</g>\n");
    if (frameid == nbframes - 1) {
        fprintf(f, "</g>\n");
        fprintf(f, "</svg>\n");
    }
    fclose(f);
}

///////////////////////// end FILE SAVING ////////////////////////////////

////////////////////// LBFGS //////////////////////

double g(std::vector<double> W, std::vector<Vector> points, std::vector<double> lambda) {
    double res = 0.;
    // compute the voronoi cells
    std::vector<Polygon> voronoi_cells = voronoi_parallel_linear_enumeration(points, W);
    // compute the integral of the norm of the difference between the voronoi cell and the pt
    for (int i = 0; i < points.size(); i++) {
        res += voronoi_cells[i].integral_norm_diff(points[i]);
        res -= W[i] * voronoi_cells[i].area();
        res += lambda[i] * W[i];
    };

    return res;
}

class objective_function {
  protected:
    lbfgsfloatval_t *m_x;
    std::vector<Vector> points;
    std::vector<lbfgsfloatval_t> lambda;

  public:
    objective_function() : m_x(NULL) {
    }

    objective_function(std::vector<Vector> points, std::vector<lbfgsfloatval_t> lambda) : m_x(NULL), points(points), lambda(lambda) {
    }

    virtual ~objective_function() {
        if (m_x != NULL) {
            lbfgs_free(m_x);
            m_x = NULL;
        }
    }

    int run(int N, std::vector<lbfgsfloatval_t> &resW) {
        lbfgsfloatval_t fx;
        lbfgsfloatval_t *m_x = lbfgs_malloc(N);

        if (m_x == NULL) {
            printf("ERROR: Failed to allocate a memory block for variables.\n");
            return 1;
        }

        /* Initialize the variables. */
        for (int i = 0; i < N; i++) {
            lbfgsfloatval_t val = rand() % 100 / 100.;
            m_x[i] = val;
        }

        /*
            Start the L-BFGS optimization; this will invoke the callback functions
            evaluate() and progress() when necessary.
         */
        int ret = lbfgs(N, m_x, &fx, _evaluate, _progress, this, NULL);

        /* Report the result. */
        printf("L-BFGS optimization terminated with status code = %d\n", ret);
        printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, m_x[0], m_x[1]);
        for (int i = 0; i < N; i++) {
            resW.push_back(m_x[i]);
        }

        return ret;
    }

  protected:
    static lbfgsfloatval_t _evaluate(
        void *instance,
        const lbfgsfloatval_t *x,
        lbfgsfloatval_t *g,
        const int n,
        const lbfgsfloatval_t step) {
        return reinterpret_cast<objective_function *>(instance)->evaluate(x, g, n, step);
    }

    lbfgsfloatval_t evaluate(
        const lbfgsfloatval_t *W,
        lbfgsfloatval_t *g,
        const int n,
        const lbfgsfloatval_t step) {

        // compute the voronoi cells

        std::vector<lbfgsfloatval_t> W_prime = std::vector<lbfgsfloatval_t>(n, 0);
        for (int i = 0; i < n; i++) {
            W_prime[i] = W[i];
        }
        std::vector<Polygon> voronoi_cells = voronoi_parallel_linear_enumeration(points, W_prime);
        save_svg(voronoi_cells, "partial.svg");
        lbfgsfloatval_t res = 0.0;
        // compute the integral of the norm of the difference between the voronoi cell and the pt
        for (int i = 0; i < points.size(); i++) {
#if DEBUG
            std::cout << i << std::endl;
            std::cout << points[i][0] << " " << points[i][1] << " " << points[i][2] << " " << std::endl;
#endif
            double integral_term = voronoi_cells[i].integral_norm_diff2(points[i]);
            double area_term = -W[i] * voronoi_cells[i].area();
            double lambda_term = lambda[i] * W[i];
            double gradient = (-voronoi_cells[i].area() + lambda[i]);
            g[i] = -gradient;
            res = res + integral_term + area_term + lambda_term;
#if DEBUG
            std::cout << "W[i] " << W[i] << std::endl;
            std::cout << "lambda[i] " << lambda[i] << std::endl;
            std::cout << "integral_term " << integral_term << std::endl;
            std::cout << "integral term 2 " << voronoi_cells[i].integral_norm_diff2(points[i]) << std::endl;
            std::cout << "area_term " << area_term << std::endl;
            std::cout << "lambda_term " << lambda_term << std::endl;
            std::cout << "-gradient " << -gradient << std::endl;
            std::cout << "cell " << std::endl;
            for (int j = 0; j < voronoi_cells[i].vertices.size(); j++) {
                std::cout << voronoi_cells[i].vertices[j][0] << " " << voronoi_cells[i].vertices[j][1] << ", ";
            }
            std::cout << std::endl;
#endif
        };
#if DEBUG
        std::cout << "-res " << -res << std::endl;
        std::cout << "___________________" << std::endl;
#endif
        return -res;
    }

    static int _progress(
        void *instance,
        const lbfgsfloatval_t *x,
        const lbfgsfloatval_t *g,
        const lbfgsfloatval_t fx,
        const lbfgsfloatval_t xnorm,
        const lbfgsfloatval_t gnorm,
        const lbfgsfloatval_t step,
        int n,
        int k,
        int ls) {
        return reinterpret_cast<objective_function *>(instance)->progress(x, g, fx, xnorm, gnorm, step, n, k, ls);
    }

    int progress(
        const lbfgsfloatval_t *x,
        const lbfgsfloatval_t *g,
        const lbfgsfloatval_t fx,
        const lbfgsfloatval_t xnorm,
        const lbfgsfloatval_t gnorm,
        const lbfgsfloatval_t step,
        int n,
        int k,
        int ls) {
#if DEBUG
        printf("Iteration %d:\n", k);
        printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, x[0], x[1]);
        printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
        printf("\n");
#endif
        return 0;
    }
};

////////////////////// end LBFGS //////////////////////

/////////////////// points generators ////////////////
std::vector<Vector> generate_points_uniform(int n) {
    std::vector<Vector> points;
    for (int i = 0; i < n; i++) {
        points.push_back(Vector((rand() % 100) / 100., (rand() % 100) / 100.));
    }
    return points;
}

#include <random>

std::vector<Vector> generate_points_normal(int n) {
    std::vector<Vector> points;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> distribution(0.5, 0.5); // mean = 0.5, standard deviation = 0.1
    double x = 0.;
    double y = 0.;
    for (int i = 0; i < n; i++) {
        do {
            x = distribution(gen);
            y = distribution(gen);
        } while (x < 0 || x > 1 || y < 0 || y > 1);
        points.push_back(Vector(x, y));
    }
    return points;
}

std::vector<Vector> generate_points_in_a_disk(int n, double r, size_t layers = 5, Vector center = Vector(0.5, 0.5)) {
    std::vector<Vector> points;
    for (size_t i = 0; i < layers; i++) {
        Polygon disk = regular_n_gon(n - n / layers * i, r - r / layers * i);
        for (Vector point : disk.vertices) {
            points.push_back(center + point);
        }
    }
    points.push_back(center);
    return points;
}

////////////////////////// end points generators //////////////////////

////////////////////////// fluid simulator //////////////////////

lbfgsfloatval_t evaluate2(
    void *instance,
    const lbfgsfloatval_t *W,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step) {

    const lbfgsfloatval_t f = 0.4; // parameter
    lbfgsfloatval_t desired_air_volume = 1 - f;
    lbfgsfloatval_t lambda = f / (n - 1);
    lbfgsfloatval_t estimated_fluid_volume = 0;
    lbfgsfloatval_t res = 0.0;

    Vector *points = static_cast<Vector *>(instance);
    std::vector<Vector> points_vec = std::vector<Vector>(points, points + n - 1);
    std::vector<double> W_vec = std::vector<double>(W, W + n - 1);

    std::vector<Polygon> cells = voronoi_parallel_linear_enumeration(points_vec, W_vec, true, W[n - 1]);

    for (int i = 0; i < n - 1; i++) {
        double integral_term = cells[i].integral_norm_diff2(points[i]);
        double area_term = -W[i] * cells[i].area();
        double lambda_term = lambda * W[i];
        double gradient = (-cells[i].area() + lambda);
        g[i] = -gradient;
        res = res + integral_term + area_term + lambda_term;
        estimated_fluid_volume += cells[i].area();
    }
    // air
    lbfgsfloatval_t estimated_air_volume = 1 - estimated_fluid_volume;
    g[n - 1] = -(desired_air_volume - estimated_air_volume);
    res += (-W[n - 1] * estimated_air_volume + desired_air_volume * W[n - 1]);

    return -res;
}
void optimize(std::vector<Vector> &points, lbfgsfloatval_t *w) {

    lbfgsfloatval_t fx = 0.;
    int ret = lbfgs(points.size() + 1, &w[0], &fx, evaluate2, nullptr, &points[0], NULL);
}

void update_positions(double epsilon, double dt, double mass, std::vector<Polygon> &cells, std::vector<Vector> &points, std::vector<Vector> &velocities) {
    for (int i = 0; i < cells.size(); ++i) {

        Vector centroid = cells[i].centroid();

        Vector spring_force = (centroid - points[i]) / pow(epsilon, 2);
        Vector weight = Vector(0, -G_FORCE) * mass;

        Vector F = weight + spring_force;

        velocities[i] = velocities[i] + dt / mass * F;
        points[i] = points[i] + dt * velocities[i];

        const double bounce_loss = 0.8; // tunable parameter
        // horizontal bounce
        if (points[i][0] < 0) {
            velocities[i][0] = -velocities[i][0] * bounce_loss;
            points[i][0] = -points[i][0];
        } else if (points[i][0] > 1) {
            velocities[i][0] = -velocities[i][0] * bounce_loss;
            points[i][0] = 1 - (points[i][0] - 1);
        }
        // vertical bounce
        if (points[i][1] < 0) {
            velocities[i][1] = -velocities[i][1] * bounce_loss;
            points[i][1] = -points[i][1];
        } else if (points[i][1] > 1) {
            velocities[i][1] = -velocities[i][1] * bounce_loss;
            points[i][1] = 1 - (points[i][1] - 1);
        }
    }
    return;
}

void simulation(size_t N, size_t n_frames) {
    std::vector<Vector> particles = generate_points_uniform(N);
    std::vector<Vector> velocities(N, Vector(0, 0));
    double m = 250;
    double eps = 0.004;
    double dt = 0.005;
    lbfgsfloatval_t *w = lbfgs_malloc(N + 1);

    for (int i = 0; i < N + 1; ++i) {
        w[i] = 1.0;
    }

    for (int t = 0; t < n_frames; t++) {
        std::cout << "computing frame " << t << std::endl;
        optimize(particles, w);
        std::vector<double> w_prime;
        for (size_t i = 0; i < N; i++) {
            w_prime.push_back(w[i]);
        }
        std::vector<Polygon> cells = voronoi_parallel_linear_enumeration(particles, w_prime, true, w[N]);
        save_frame(cells, "frame_", t);
        update_positions(eps, dt, m, cells, particles, velocities);
    }
    lbfgs_free(w);
}

////////////////////////// end fluid //////////////////////

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    //simulation(40, 200);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Took  " << duration.count() << "ms." << std::endl;

    return 0;
}

// :)