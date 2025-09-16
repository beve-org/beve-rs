use serde::{ser, Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex<T> {
    pub re: T,
    pub im: T,
}

impl Serialize for Complex<f32> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut tup = s.serialize_tuple(2)?;
        ser::SerializeTuple::serialize_element(&mut tup, &self.re)?;
        ser::SerializeTuple::serialize_element(&mut tup, &self.im)?;
        ser::SerializeTuple::end(tup)
    }
}

impl Serialize for Complex<f64> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut tup = s.serialize_tuple(2)?;
        ser::SerializeTuple::serialize_element(&mut tup, &self.re)?;
        ser::SerializeTuple::serialize_element(&mut tup, &self.im)?;
        ser::SerializeTuple::end(tup)
    }
}

impl<'de, T> serde::Deserialize<'de> for Complex<T>
where
    T: serde::Deserialize<'de>,
{
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V<T>(core::marker::PhantomData<T>);
        impl<'de, T> serde::de::Visitor<'de> for V<T>
        where
            T: serde::Deserialize<'de>,
        {
            type Value = Complex<T>;
            fn expecting(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                write!(f, "complex number as [re, im]")
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut a: A,
            ) -> Result<Self::Value, A::Error> {
                let re: T = a
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::custom("missing real"))?;
                let im: T = a
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::custom("missing imag"))?;
                Ok(Complex { re, im })
            }
        }
        d.deserialize_seq(V(core::marker::PhantomData))
    }
}

pub struct ComplexSlice<'a, T>(pub &'a [Complex<T>]);

impl<'a> Serialize for ComplexSlice<'a, f32> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(self.0.len()))?;
        for c in self.0 {
            ser::SerializeSeq::serialize_element(&mut seq, &(c.re, c.im))?;
        }
        ser::SerializeSeq::end(seq)
    }
}

impl<'a> Serialize for ComplexSlice<'a, f64> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(self.0.len()))?;
        for c in self.0 {
            ser::SerializeSeq::serialize_element(&mut seq, &(c.re, c.im))?;
        }
        ser::SerializeSeq::end(seq)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixLayout {
    Right,
    Left,
}

impl<'de> Deserialize<'de> for MatrixLayout {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = MatrixLayout;
            fn expecting(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                write!(f, "matrix layout string")
            }
            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
                match v {
                    "layout_right" | "row_major" | "right" => Ok(MatrixLayout::Right),
                    "layout_left" | "column_major" | "left" => Ok(MatrixLayout::Left),
                    _ => Err(E::custom("invalid matrix layout")),
                }
            }
        }
        d.deserialize_str(V)
    }
}

pub struct Matrix<'a, T> {
    pub layout: MatrixLayout,
    pub extents: &'a [usize],
    pub data: &'a [T],
}

impl<'a, T: Serialize> Serialize for Matrix<'a, T> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        // Fallback serde map: { layout: ..., extents: [...], value: [...] }
        let mut map = s.serialize_map(Some(3))?;
        ser::SerializeMap::serialize_entry(
            &mut map,
            "layout",
            &match self.layout {
                MatrixLayout::Right => "layout_right",
                MatrixLayout::Left => "layout_left",
            },
        )?;
        ser::SerializeMap::serialize_entry(&mut map, "extents", &self.extents)?;
        ser::SerializeMap::serialize_entry(&mut map, "value", &self.data)?;
        ser::SerializeMap::end(map)
    }
}
