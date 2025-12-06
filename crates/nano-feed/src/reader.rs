//! File reader for MDP 3.0 data files.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use crate::error::{FeedError, FeedResult};
use crate::messages::MdpMessage;
use crate::parser::MdpParser;

/// Reader for MDP 3.0 data files
pub struct MdpFileReader {
    reader: BufReader<File>,
    parser: MdpParser,
    buffer: Vec<u8>,
    buffer_pos: usize,
    buffer_len: usize,
}

impl MdpFileReader {
    /// Buffer size for file reading (1MB)
    const BUFFER_SIZE: usize = 1024 * 1024;

    /// Open a file for reading
    pub fn open<P: AsRef<Path>>(path: P) -> FeedResult<Self> {
        let file = File::open(path)?;
        Ok(Self {
            reader: BufReader::with_capacity(Self::BUFFER_SIZE, file),
            parser: MdpParser::new(),
            buffer: vec![0u8; Self::BUFFER_SIZE * 2],
            buffer_pos: 0,
            buffer_len: 0,
        })
    }

    /// Read the next message from the file
    pub fn next_message(&mut self) -> FeedResult<Option<MdpMessage>> {
        loop {
            // Try to parse from existing buffer
            if self.buffer_pos < self.buffer_len {
                let data = &self.buffer[self.buffer_pos..self.buffer_len];
                match self.parser.parse(data) {
                    Ok((msg, remaining)) => {
                        self.buffer_pos = self.buffer_len - remaining.len();
                        return Ok(Some(msg));
                    }
                    Err(FeedError::Incomplete { .. }) => {
                        // Need more data, fall through to read more
                    }
                    Err(e) => return Err(e),
                }
            }

            // Shift remaining data to start of buffer
            if self.buffer_pos > 0 && self.buffer_pos < self.buffer_len {
                let remaining = self.buffer_len - self.buffer_pos;
                self.buffer.copy_within(self.buffer_pos..self.buffer_len, 0);
                self.buffer_pos = 0;
                self.buffer_len = remaining;
            } else {
                self.buffer_pos = 0;
                self.buffer_len = 0;
            }

            // Read more data
            let bytes_read = self.reader.read(&mut self.buffer[self.buffer_len..])?;
            if bytes_read == 0 {
                if self.buffer_len > self.buffer_pos {
                    // Incomplete message at end of file
                    return Err(FeedError::Incomplete {
                        needed: 1,
                    });
                }
                return Ok(None); // EOF
            }
            self.buffer_len += bytes_read;
        }
    }

    /// Create an iterator over messages
    pub fn messages(self) -> MdpMessageIterator {
        MdpMessageIterator { reader: self }
    }

    /// Reset the parser state
    pub fn reset_parser(&mut self) {
        self.parser.reset();
    }

    /// Get current sequence number
    pub fn current_sequence(&self) -> u32 {
        self.parser.expected_sequence()
    }
}

/// Iterator over MDP messages
pub struct MdpMessageIterator {
    reader: MdpFileReader,
}

impl Iterator for MdpMessageIterator {
    type Item = FeedResult<MdpMessage>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.reader.next_message() {
            Ok(Some(msg)) => Some(Ok(msg)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

/// Memory-mapped file reader for maximum performance
#[cfg(feature = "mmap")]
pub struct MdpMmapReader {
    mmap: memmap2::Mmap,
    parser: MdpParser,
    position: usize,
}

#[cfg(feature = "mmap")]
impl MdpMmapReader {
    /// Open a file with memory mapping
    pub fn open<P: AsRef<Path>>(path: P) -> FeedResult<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        Ok(Self {
            mmap,
            parser: MdpParser::new(),
            position: 0,
        })
    }

    /// Read the next message
    pub fn next_message(&mut self) -> FeedResult<Option<MdpMessage>> {
        if self.position >= self.mmap.len() {
            return Ok(None);
        }

        let data = &self.mmap[self.position..];
        match self.parser.parse(data) {
            Ok((msg, remaining)) => {
                self.position = self.mmap.len() - remaining.len();
                Ok(Some(msg))
            }
            Err(e) => Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    // Tests would require test data files
}

